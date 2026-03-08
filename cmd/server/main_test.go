package main

import (
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"testing"
)

func TestIsMultipartFormRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		contentType string
		want        bool
	}{
		{name: "multipart", contentType: "multipart/form-data; boundary=abc123", want: true},
		{name: "wrong type", contentType: "application/json", want: false},
		{name: "invalid", contentType: "%%%invalid%%%", want: false},
		{name: "empty", contentType: "", want: false},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := isMultipartFormRequest(tc.contentType); got != tc.want {
				t.Fatalf("isMultipartFormRequest(%q) = %v, want %v", tc.contentType, got, tc.want)
			}
		})
	}
}

func TestValidateUploadedFile(t *testing.T) {
	t.Parallel()

	makeHeader := func(filename string, size int64) *multipart.FileHeader {
		return &multipart.FileHeader{
			Filename: filename,
			Size:     size,
			Header:   textproto.MIMEHeader{"Content-Type": []string{"image/jpeg"}},
		}
	}

	tests := []struct {
		name    string
		header  *multipart.FileHeader
		maxSize int64
		wantErr bool
	}{
		{name: "valid", header: makeHeader("sample.jpg", 1024), maxSize: 2048, wantErr: false},
		{name: "nil", header: nil, maxSize: 2048, wantErr: true},
		{name: "empty filename", header: makeHeader("", 1024), maxSize: 2048, wantErr: true},
		{name: "empty file", header: makeHeader("sample.jpg", 0), maxSize: 2048, wantErr: true},
		{name: "too large", header: makeHeader("sample.jpg", 4096), maxSize: 2048, wantErr: true},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := validateUploadedFile(tc.header, tc.maxSize)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateUploadedFile() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestNewServerAppliesMinimums(t *testing.T) {
	t.Parallel()

	s := newServer(nil, 0, 0, 0, 0, 0)
	if cap(s.inflightSem) != 1 {
		t.Fatalf("inflight cap = %d, want 1", cap(s.inflightSem))
	}
	if s.maxUploadBytes != 32*1024*1024 {
		t.Fatalf("maxUploadBytes = %d, want %d", s.maxUploadBytes, 32*1024*1024)
	}
	if s.maxFileBytes != 32*1024*1024 {
		t.Fatalf("maxFileBytes = %d, want %d", s.maxFileBytes, 32*1024*1024)
	}
	if s.maxFiles != 1 {
		t.Fatalf("maxFiles = %d, want 1", s.maxFiles)
	}
	if s.maxLimit != 50 {
		t.Fatalf("maxLimit = %d, want 50", s.maxLimit)
	}
}

func TestHandleEvaluateRejectsWrongContentType(t *testing.T) {
	t.Parallel()

	s := newServer(nil, 1, 32, 16, 8, 200)
	req, err := http.NewRequest(http.MethodPost, "/evaluate", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	s.handleEvaluate(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
}
