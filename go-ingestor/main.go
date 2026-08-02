// Command go-ingestor polls a financial news REST API
// (or a local JSON fixture when no API key is configured) and forwards
// normalized articles to the Python agent's /ingest endpoint.
//
// Offline demo mode: pass -sample-file path/to/fixture.json — no API key needed.
// Live mode: set FINANCIAL_NEWS_API_KEY and the ingestor calls the configured endpoint.
// In production this would consume a WebSocket/TCP streaming feed instead of polling REST
// on an interval; polling is used here to keep the demo dependency-free.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// NewsDoc mirrors the schema expected by the Python RAG service's
// /ingest endpoint (see python-rag/store.py NewsDoc).
type NewsDoc struct {
	ID        string `json:"id"`
	Ticker    string `json:"ticker"`
	Timestamp string `json:"timestamp"`
	Channel   string `json:"channel"`
	Headline  string `json:"headline"`
	Body      string `json:"body"`
}

// newsAPIRawItem models the subset of fields we care about from
// the financial news API response.
type newsAPIRawItem struct {
	ID      int    `json:"id"`
	Created string `json:"created"`
	Title   string `json:"title"`
	Body    string `json:"body"`
	Teaser  string `json:"teaser"`
	Stocks  []struct {
		Name string `json:"name"`
	} `json:"stocks"`
	Channels []struct {
		Name string `json:"name"`
	} `json:"channels"`
}

// newsAPIURL is the financial news REST endpoint.
// Configure via NEWS_API_URL env var (defaults to the demo stub).
var newsAPIURL = func() string {
	if u := os.Getenv("NEWS_API_URL"); u != "" {
		return u
	}
	return "https://api.example.com/v2/news"
}()

func main() {
	var (
		ragURL     = flag.String("rag-url", "http://localhost:8000/ingest", "Python RAG service ingest endpoint")
		interval   = flag.Duration("interval", 30*time.Second, "polling interval")
		sampleFile = flag.String("sample-file", "", "path to a local JSON fixture to use instead of the live API (offline demo mode)")
		tickers    = flag.String("tickers", "", "comma-separated tickers to filter on (live API mode only)")
		once       = flag.Bool("once", false, "run a single ingest pass and exit (useful for testing)")
	)
	flag.Parse()

	apiKey := os.Getenv("FINANCIAL_NEWS_API_KEY")

	for {
		var docs []NewsDoc
		var err error

		switch {
		case *sampleFile != "":
			docs, err = loadSampleFile(*sampleFile)
		case apiKey != "":
			docs, err = fetchLiveNews(apiKey, *tickers)
		default:
			err = fmt.Errorf("no FINANCIAL_NEWS_API_KEY set and no -sample-file provided; nothing to ingest")
		}

		if err != nil {
			log.Printf("fetch error: %v", err)
		} else if len(docs) > 0 {
			if err := pushToRAG(*ragURL, docs); err != nil {
				log.Printf("push error: %v", err)
			} else {
				log.Printf("ingested %d docs -> %s", len(docs), *ragURL)
			}
		}

		if *once {
			return
		}
		time.Sleep(*interval)
	}
}

func loadSampleFile(path string) ([]NewsDoc, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading sample file: %w", err)
	}
	var docs []NewsDoc
	if err := json.Unmarshal(data, &docs); err != nil {
		return nil, fmt.Errorf("parsing sample file: %w", err)
	}
	return docs, nil
}

func fetchLiveNews(apiKey, tickers string) ([]NewsDoc, error) {
	url := fmt.Sprintf("%s?token=%s&pageSize=20&displayOutput=full", newsAPIURL, apiKey)
	if tickers != "" {
		url += "&tickers=" + tickers
	}

	resp, err := http.Get(url) //nolint:noctx
	if err != nil {
		return nil, fmt.Errorf("calling news API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("news API returned %d: %s", resp.StatusCode, string(body))
	}

	var raw []newsAPIRawItem
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	docs := make([]NewsDoc, 0, len(raw))
	for _, item := range raw {
		ticker := "N/A"
		if len(item.Stocks) > 0 {
			ticker = item.Stocks[0].Name
		}
		channel := "news"
		if len(item.Channels) > 0 {
			channel = item.Channels[0].Name
		}
		body := item.Body
		if body == "" {
			body = item.Teaser
		}
		docs = append(docs, NewsDoc{
			ID:        fmt.Sprintf("%d", item.ID),
			Ticker:    ticker,
			Timestamp: item.Created,
			Channel:   channel,
			Headline:  item.Title,
			Body:      body,
		})
	}
	return docs, nil
}

func pushToRAG(ragURL string, docs []NewsDoc) error {
	payload, err := json.Marshal(docs)
	if err != nil {
		return fmt.Errorf("marshaling docs: %w", err)
	}

	resp, err := http.Post(ragURL, "application/json", bytes.NewReader(payload)) //nolint:noctx
	if err != nil {
		return fmt.Errorf("posting to RAG service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("RAG service returned %d: %s", resp.StatusCode, string(body))
	}
	return nil
}
