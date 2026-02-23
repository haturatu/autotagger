package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type prediction struct {
	Filename string             `json:"filename"`
	Tags     map[string]float64 `json:"tags"`
}

type workerRequest struct {
	ID        uint64   `json:"id"`
	Files     []string `json:"files"`
	Threshold float64  `json:"threshold"`
	Limit     int      `json:"limit"`
}

type workerResponse struct {
	ID          uint64       `json:"id"`
	Predictions []prediction `json:"predictions,omitempty"`
	Error       string       `json:"error,omitempty"`
}

type workerClient struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	pending map[uint64]chan workerResponse

	pendingMu sync.Mutex
	writeMu   sync.Mutex
	nextID    atomic.Uint64
	closed    atomic.Bool
}

func newWorkerClient(ctx context.Context, pythonBin, scriptPath string) (*workerClient, error) {
	cmd := exec.CommandContext(ctx, pythonBin, scriptPath)
	cmd.Env = os.Environ()

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("open worker stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("open worker stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("open worker stderr: %w", err)
	}

	wc := &workerClient{
		cmd:     cmd,
		stdin:   stdin,
		pending: make(map[uint64]chan workerResponse),
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start worker: %w", err)
	}

	go wc.readStdout(stdout)
	go wc.readStderr(stderr)
	go wc.waitProcess()

	return wc, nil
}

func (wc *workerClient) readStdout(r io.Reader) {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 16*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		var resp workerResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			slog.Error("worker invalid response", "error", err)
			continue
		}

		wc.pendingMu.Lock()
		ch, ok := wc.pending[resp.ID]
		if ok {
			delete(wc.pending, resp.ID)
		}
		wc.pendingMu.Unlock()
		if ok {
			ch <- resp
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("worker stdout error", "error", err)
	}
	wc.failAll("worker stdout closed")
}

func (wc *workerClient) readStderr(r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		slog.Info("worker stderr", "line", scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		slog.Error("worker stderr error", "error", err)
	}
}

func (wc *workerClient) waitProcess() {
	if err := wc.cmd.Wait(); err != nil {
		slog.Error("worker exited", "error", err)
	}
	wc.closed.Store(true)
	wc.failAll("worker exited")
}

func (wc *workerClient) failAll(msg string) {
	wc.pendingMu.Lock()
	defer wc.pendingMu.Unlock()
	for id, ch := range wc.pending {
		delete(wc.pending, id)
		ch <- workerResponse{ID: id, Error: msg}
	}
}

func (wc *workerClient) predict(ctx context.Context, files []string, threshold float64, limit int) ([]prediction, error) {
	if wc.closed.Load() {
		return nil, errors.New("worker is not running")
	}

	id := wc.nextID.Add(1)
	respCh := make(chan workerResponse, 1)
	wc.pendingMu.Lock()
	wc.pending[id] = respCh
	wc.pendingMu.Unlock()

	req := workerRequest{ID: id, Files: files, Threshold: threshold, Limit: limit}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	wc.writeMu.Lock()
	_, err = wc.stdin.Write(append(data, '\n'))
	wc.writeMu.Unlock()
	if err != nil {
		wc.pendingMu.Lock()
		delete(wc.pending, id)
		wc.pendingMu.Unlock()
		return nil, fmt.Errorf("write request: %w", err)
	}

	select {
	case resp := <-respCh:
		if resp.Error != "" {
			return nil, errors.New(resp.Error)
		}
		return resp.Predictions, nil
	case <-ctx.Done():
		wc.pendingMu.Lock()
		delete(wc.pending, id)
		wc.pendingMu.Unlock()
		return nil, ctx.Err()
	}
}

func (wc *workerClient) close() {
	if wc.closed.Swap(true) {
		return
	}
	_ = wc.stdin.Close()
	if wc.cmd.Process != nil {
		_ = wc.cmd.Process.Signal(syscall.SIGTERM)
	}
}

type tagPair struct {
	Name  string
	Score float64
}

type htmlResult struct {
	ImageData string
	Tags      []tagPair
	TagText   string
}

type server struct {
	worker         *workerClient
	inflightSem    chan struct{}
	maxUploadBytes int64
	evaluateOK     atomic.Bool
	fatalOnce      sync.Once
	indexTmpl      *template.Template
	evalTmpl       *template.Template
	errorTmpl      *template.Template
}

func newServer(worker *workerClient, maxInflight int, maxUploadMB int64) *server {
	if maxInflight < 1 {
		maxInflight = 1
	}
	if maxUploadMB < 1 {
		maxUploadMB = 32
	}
	s := &server{
		worker:         worker,
		inflightSem:    make(chan struct{}, maxInflight),
		maxUploadBytes: maxUploadMB * 1024 * 1024,
		indexTmpl:      template.Must(template.New("index").Parse(indexHTML)),
		evalTmpl: template.Must(template.New("evaluate").Funcs(template.FuncMap{
			"mul100": func(v float64) float64 { return v * 100 },
		}).Parse(evaluateHTML)),
		errorTmpl: template.Must(template.New("error").Parse(errorHTML)),
	}
	s.evaluateOK.Store(true)
	return s
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/evaluate", s.handleEvaluate)
	mux.HandleFunc("/healthz", s.handleHealth)
	return s.loggingMiddleware(mux)
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (sr *statusRecorder) WriteHeader(status int) {
	sr.status = status
	sr.ResponseWriter.WriteHeader(status)
}

func (sr *statusRecorder) Write(b []byte) (int, error) {
	if sr.status == 0 {
		sr.status = http.StatusOK
	}
	n, err := sr.ResponseWriter.Write(b)
	sr.bytes += n
	return n, err
}

func (s *server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)

		if rec.status == 0 {
			rec.status = http.StatusOK
		}
		slog.Info("http_request",
			"method", r.Method,
			"path", r.URL.Path,
			"query", r.URL.RawQuery,
			"status", rec.status,
			"bytes", rec.bytes,
			"latency_ms", time.Since(start).Milliseconds(),
			"remote_addr", r.RemoteAddr,
			"user_agent", r.UserAgent(),
		)
	})
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if s.worker.closed.Load() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "worker_down"})
		return
	}
	if !s.evaluateOK.Load() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "evaluate_error"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := s.indexTmpl.Execute(w, nil); err != nil {
		slog.Error("render index failed", "error", err)
	}
}

func (s *server) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.inflightSem <- struct{}{}
	defer func() { <-s.inflightSem }()

	format := "html"

	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadBytes)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		s.writeError(w, format, http.StatusBadRequest, "BadRequest", "invalid multipart body or request too large")
		return
	}
	if f := strings.ToLower(strings.TrimSpace(r.FormValue("format"))); f != "" {
		format = f
	}

	threshold, err := parseFloatOrDefault(r.FormValue("threshold"), 0.1)
	if err != nil {
		s.writeError(w, format, http.StatusBadRequest, "BadRequest", "threshold must be a float")
		return
	}
	limit, err := parseIntOrDefault(r.FormValue("limit"), 50)
	if err != nil || limit < 1 {
		s.writeError(w, format, http.StatusBadRequest, "BadRequest", "limit must be a positive integer")
		return
	}

	files := r.MultipartForm.File["file"]
	if len(files) == 0 {
		s.writeError(w, format, http.StatusBadRequest, "BadRequest", "at least one file is required")
		return
	}

	tmpDir, err := os.MkdirTemp("", "autotagger-upload-*")
	if err != nil {
		s.writeError(w, format, http.StatusInternalServerError, "InternalError", "failed to create temp dir")
		return
	}
	defer os.RemoveAll(tmpDir)

	paths := make([]string, 0, len(files))
	origNames := make([]string, 0, len(files))
	for i, fh := range files {
		f, err := fh.Open()
		if err != nil {
			s.writeError(w, format, http.StatusBadRequest, "BadRequest", "failed to open upload")
			return
		}

		safeName := sanitizeFilename(fh.Filename, i)
		dstPath := filepath.Join(tmpDir, safeName)
		dst, err := os.Create(dstPath)
		if err != nil {
			_ = f.Close()
			s.writeError(w, format, http.StatusInternalServerError, "InternalError", "failed to store upload")
			return
		}

		_, copyErr := io.Copy(dst, f)
		_ = dst.Close()
		_ = f.Close()
		if copyErr != nil {
			s.writeError(w, format, http.StatusInternalServerError, "InternalError", "failed to read upload")
			return
		}

		paths = append(paths, dstPath)
		origNames = append(origNames, fh.Filename)
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	predictions, err := s.worker.predict(ctx, paths, threshold, limit)
	if err != nil {
		slog.Error("predict failed", "error", err)
		s.writeError(w, format, http.StatusInternalServerError, "InferenceError", err.Error())
		return
	}

	for i := range predictions {
		if i < len(origNames) {
			predictions[i].Filename = origNames[i]
		}
	}
	s.evaluateOK.Store(true)

	switch format {
	case "json":
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(predictions); err != nil {
			slog.Error("encode json failed", "error", err)
		}
	case "html":
		results, err := buildHTMLResults(paths, predictions)
		if err != nil {
			s.writeError(w, format, http.StatusInternalServerError, "InternalError", "failed to render HTML")
			return
		}
		if err := s.evalTmpl.Execute(w, results); err != nil {
			slog.Error("render evaluate failed", "error", err)
		}
	default:
		s.writeError(w, format, http.StatusBadRequest, "BadRequest", "format must be html or json")
	}
}

func buildHTMLResults(paths []string, predictions []prediction) ([]htmlResult, error) {
	results := make([]htmlResult, 0, len(predictions))
	for i, pred := range predictions {
		if i >= len(paths) {
			break
		}
		data, err := os.ReadFile(paths[i])
		if err != nil {
			return nil, err
		}
		tags := make([]tagPair, 0, len(pred.Tags))
		tagNames := make([]string, 0, len(pred.Tags))
		for name, score := range pred.Tags {
			tags = append(tags, tagPair{Name: name, Score: score})
			tagNames = append(tagNames, name)
		}
		sort.Slice(tags, func(a, b int) bool {
			return tags[a].Score > tags[b].Score
		})
		sort.Strings(tagNames)

		results = append(results, htmlResult{
			ImageData: base64.StdEncoding.EncodeToString(data),
			Tags:      tags,
			TagText:   strings.Join(tagNames, " "),
		})
	}
	return results, nil
}

func (s *server) writeError(w http.ResponseWriter, format string, status int, errName, message string) {
	if status >= 500 {
		s.evaluateOK.Store(false)
	}
	if format == "json" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   errName,
			"message": message,
		})
	} else {
		w.WriteHeader(status)
		_ = s.errorTmpl.Execute(w, map[string]string{"Error": errName, "Message": message})
	}

	if status >= 500 {
		s.fatalOnce.Do(func() {
			slog.Error("fatal evaluate error detected; exiting process for container restart", "status", status, "error", errName)
			go func() {
				time.Sleep(100 * time.Millisecond)
				os.Exit(1)
			}()
		})
	}
}

func parseFloatOrDefault(raw string, def float64) (float64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return def, nil
	}
	return strconv.ParseFloat(raw, 64)
}

func parseIntOrDefault(raw string, def int) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return def, nil
	}
	return strconv.Atoi(raw)
}

func getenvInt(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func getenvInt64(key string, def int64) int64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return def
	}
	return n
}

func sanitizeFilename(name string, index int) string {
	base := filepath.Base(strings.TrimSpace(name))
	if base == "" || base == "." || base == string(filepath.Separator) {
		return fmt.Sprintf("upload-%d", index)
	}
	base = strings.ReplaceAll(base, "..", "")
	return base
}

func main() {
	logLevel := strings.ToLower(strings.TrimSpace(os.Getenv("LOG_LEVEL")))
	if logLevel == "" {
		logLevel = "info"
	}
	var level slog.Level
	switch logLevel {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})))

	addr := strings.TrimSpace(os.Getenv("HTTP_ADDR"))
	if addr == "" {
		addr = ":5000"
	}

	pythonBin := strings.TrimSpace(os.Getenv("PYTHON_BIN"))
	if pythonBin == "" {
		pythonBin = "python"
	}

	scriptPath := strings.TrimSpace(os.Getenv("WORKER_SCRIPT"))
	if scriptPath == "" {
		scriptPath = "./inference_worker.py"
	}

	maxInflight := getenvInt("MAX_INFLIGHT", 2)
	maxUploadMB := getenvInt64("MAX_UPLOAD_MB", 32)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	worker, err := newWorkerClient(ctx, pythonBin, scriptPath)
	if err != nil {
		slog.Error("start worker failed", "error", err)
		os.Exit(1)
	}
	defer worker.close()

	srv := &http.Server{
		Addr:              addr,
		Handler:           newServer(worker, maxInflight, maxUploadMB).routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      6 * time.Minute,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			slog.Error("server shutdown failed", "error", err)
		}
	}()

	slog.Info(
		"server listening",
		"addr", addr,
		"max_inflight", maxInflight,
		"max_upload_mb", maxUploadMB,
	)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}
}

const indexHTML = `<!DOCTYPE html>
<html>
  <body>
    <form action="/evaluate" method="post" enctype="multipart/form-data">
      <input type="file" name="file" multiple>
      <input type="hidden" name="threshold" min="0" max="1" step="0.1" value="0.01">
      <input type="hidden" name="limit" value="100">
      <input type="submit" value="Submit">
    </form>
  </body>
</html>`

const evaluateHTML = `<!DOCTYPE html>
<html>
  <head>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>

  <body class="text-sm m-4 break-all lg:max-w-[960px] lg:mx-auto" style="font-family: system-ui;">
    <h1 class="text-3xl">Results</h1>
    <a class="text-xs text-sky-600 hover:text-sky-500 mr-4" href="/">&lt; Back</a>

    <div class="mt-4">
      {{ range . }}
        <div class="flex flex-col p-2 gap-2 border rounded md:flex-row md:max-h-[80vh]">
          <div class="flex-1 flex items-center justify-center">
            <img class="max-w-full max-h-full h-auto" src="data:image/jpg;base64,{{ .ImageData }}">
          </div>

          <div class="flex-0 overflow-scroll md:pr-2">
            <table class="w-full leading-4">
              {{ range .Tags }}
              <tr>
                <td>
                  <a class="text-sky-600 hover:text-sky-500" href="https://danbooru.donmai.us/wiki_pages/{{ .Name }}">?</a>
                  <a class="text-sky-600 hover:text-sky-500 mr-4" href="https://danbooru.donmai.us/posts?tags={{ .Name }}">{{ .Name }}</a>
                </td>
                <td class="text-gray-400 text-right">{{ printf "%.0f%%" (mul100 .Score) }}</td>
              </tr>
              {{ end }}
            </table>

            <textarea class="w-full text-gray-500 mt-2" rows="4">{{ .TagText }}</textarea>
          </div>
        </div>
      {{ end }}
    </div>
  </body>
</html>`

const errorHTML = `<!DOCTYPE html>
<html>
  <head>
    <title>{{ .Error }}</title>
  </head>
  <body>
      <h1>{{ .Error }}</h1>
      <p>{{ .Message }}</p>
  </body>
</html>`
