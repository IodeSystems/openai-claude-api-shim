package shim

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func userParts(parts string) string {
	return `{"model":"m","messages":[{"role":"user","content":[` + parts + `]}]}`
}

func TestRequestPDFFileIsABase64Document(t *testing.T) {
	data := b64("%PDF-1.4 fake")
	req := translate(t, userParts(`{"type":"text","text":"read"},
		{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,`+data+`"},"cache_control":{"type":"ephemeral"}}`), Options{})
	c := req.Messages[0].Content
	if len(c) != 2 {
		t.Fatalf("content = %+v", c)
	}
	d := c[1]
	if d.Type != "document" || d.Title != "a.pdf" || d.Source == nil ||
		d.Source.Type != "base64" || d.Source.MediaType != "application/pdf" || d.Source.Data != data {
		t.Errorf("document = %+v %+v", d, d.Source)
	}
	if len(d.CacheControl) == 0 {
		t.Errorf("cache_control on the file part was dropped")
	}
}

func TestRequestFlatInputFileIsADocument(t *testing.T) {
	req := translate(t, userParts(`{"type":"input_file","filename":"b.pdf","file_data":"data:application/pdf;base64,`+b64("x")+`"}`), Options{})
	if d := req.Messages[0].Content[0]; d.Type != "document" || d.Title != "b.pdf" {
		t.Errorf("document = %+v", d)
	}
}

func TestRequestTextFileIsDecodedToAPlainTextDocument(t *testing.T) {
	for _, mime := range []string{"text/plain", "text/markdown", "application/json"} {
		req := translate(t, userParts(`{"type":"file","file":{"filename":"n","file_data":"data:`+mime+`;base64,`+b64("héllo")+`"}}`), Options{})
		d := req.Messages[0].Content[0]
		if d.Type != "document" || d.Source == nil || d.Source.Type != "text" ||
			d.Source.MediaType != "text/plain" || d.Source.Data != "héllo" {
			t.Errorf("%s: document = %+v %+v", mime, d, d.Source)
		}
	}
}

func TestRequestDocumentTitleIsCappedAndOmittedWhenEmpty(t *testing.T) {
	long := strings.Repeat("é", 600)
	req := translate(t, userParts(`{"type":"file","file":{"filename":"`+long+`","file_data":"data:application/pdf;base64,eA=="}},
		{"type":"file","file":{"file_data":"data:application/pdf;base64,eA=="}}`), Options{})
	if n := len([]rune(req.Messages[0].Content[0].Title)); n != maxTitle {
		t.Errorf("title length = %d, want %d", n, maxTitle)
	}
	out, _ := Request([]byte(userParts(`{"type":"file","file":{"file_data":"data:application/pdf;base64,eA=="}}`)), Options{})
	if strings.Contains(string(out), `"title"`) {
		t.Errorf("empty title must be omitted (the API's minLength is 1): %s", out)
	}
}

func TestRequestRefusesUnsupportedMediaWithTheSentinel(t *testing.T) {
	cases := map[string]string{
		"input_audio":  `{"type":"input_audio","input_audio":{"data":"` + b64("RIFF") + `","format":"wav"}}`,
		"audio_url":    `{"type":"audio_url","audio_url":{"url":"https://x/a.mp3"}}`,
		"video_url":    `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}`,
		"input_video":  `{"type":"input_video","input_video":{"data":"AAAA"}}`,
		"file_id":      `{"type":"file","file":{"file_id":"file-abc"}}`,
		"file by url":  `{"type":"file","file":{"file_data":"https://x/a.pdf"}}`,
		"docx":         `{"type":"file","file":{"file_data":"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,AAAA"}}`,
		"unknown part": `{"type":"hologram"}`,
		"ftp image":    `{"type":"image_url","image_url":{"url":"ftp://x/y.png"}}`,
	}
	for name, part := range cases {
		_, err := Request([]byte(userParts(part)), Options{})
		if !errors.Is(err, ErrUnsupportedContent) {
			t.Errorf("%s: err = %v, want ErrUnsupportedContent", name, err)
		}
	}
	_, err := Request([]byte(userParts(cases["input_audio"])), Options{})
	if err == nil || !strings.Contains(err.Error(), "transcribe") {
		t.Errorf("audio refusal should say what to do instead: %v", err)
	}
}

func TestRequestRefusesMalformedFiles(t *testing.T) {
	cases := map[string]string{
		"not base64 text": `{"type":"file","file":{"file_data":"data:text/plain;base64,!!!"}}`,
		"not utf-8":       `{"type":"file","file":{"file_data":"data:text/plain;base64,` + base64.StdEncoding.EncodeToString([]byte{0xff, 0xfe}) + `"}}`,
		"no base64 flag":  `{"type":"file","file":{"file_data":"data:application/pdf,xyz"}}`,
		"in system":       ``,
	}
	cases["in system"] = `{"model":"m","messages":[{"role":"system","content":[{"type":"file","file":{"file_data":"data:application/pdf;base64,eA=="}}]}]}`
	for name, part := range cases {
		body := part
		if name != "in system" {
			body = userParts(part)
		}
		if _, err := Request([]byte(body), Options{}); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}

func TestReversePDFDocumentIsAFilePartAndRoundTrips(t *testing.T) {
	data := b64("%PDF-1.4 fake")
	out := reversed(t, `{"model":"m","max_tokens":9,"messages":[{"role":"user","content":[
		{"type":"document","title":"a.pdf","source":{"type":"base64","media_type":"application/pdf","data":"`+data+`"}},
		{"type":"text","text":"read"}]}]}`)
	msg := out["messages"].([]any)[0].(map[string]any)
	part := msg["content"].([]any)[0].(map[string]any)
	f, _ := part["file"].(map[string]any)
	if part["type"] != "file" || f["filename"] != "a.pdf" || f["file_data"] != "data:application/pdf;base64,"+data {
		t.Fatalf("file part = %v", part)
	}
	// And back: Request reads what ReverseRequest wrote.
	b, _ := json.Marshal(out)
	req := translate(t, string(b), Options{})
	d := req.Messages[0].Content[0]
	if d.Type != "document" || d.Title != "a.pdf" || d.Source.MediaType != "application/pdf" || d.Source.Data != data {
		t.Errorf("round trip = %+v %+v", d, d.Source)
	}
}

func TestReverseTextDocumentIsText(t *testing.T) {
	out := reversed(t, `{"model":"m","max_tokens":9,"messages":[{"role":"user","content":[
		{"type":"document","source":{"type":"text","media_type":"text/plain","data":"notes"}}]}]}`)
	if c := out["messages"].([]any)[0].(map[string]any)["content"]; c != "notes" {
		t.Errorf("content = %v", c)
	}
}

func TestReverseRefusesDocumentsWithNoOpenAIForm(t *testing.T) {
	for name, src := range map[string]string{
		"url":     `{"type":"url","url":"https://x/a.pdf"}`,
		"file id": `{"type":"file","file_id":"f"}`,
	} {
		_, err := ReverseRequest([]byte(`{"model":"m","max_tokens":9,"messages":[{"role":"user","content":[{"type":"document","source":` + src + `}]}]}`))
		if err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
