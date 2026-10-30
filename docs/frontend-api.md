# Local Rag API

This is the contract for the React app. The server answers questions from the company PDFs. Retrieval and the language model both run on the machine that hosts this API. The browser does not need an API key.

## Run the server

On the machine that has the PDFs, the search index, and Ollama:

```bash
uv run python -m api.server --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` lets a React dev server on another computer reach it. From the same computer, the base URL is:

```text
http://127.0.0.1:8000
```

From another computer, use the host's LAN address, for example `http://192.168.1.20:8000`.

CORS allows any origin, so the Vite or Create React App dev server can call it directly. There is no cookie and no auth header.

Open `http://127.0.0.1:8000/docs` for the generated schema. `GET /health` returns `{"status":"ok",...}` when the model is loaded.

The first token still takes a few seconds. Search runs before the model starts writing. After that, streaming shows the answer as it is produced. A warm answer is often about 3 to 8 seconds end to end. The server runs about 2 answers at a time. If 30 more are already waiting, the next call gets HTTP 429.

## Ask for the whole answer

`POST /api/ask`

`POST /ask` is the same handler. Use `/api/ask` from React.

### Request

```http
POST /api/ask
Content-Type: application/json
```

```json
{
  "question": "How many privilege leave days does an O1 employee in Nashik get?",
  "history": []
}
```

`history` is the earlier turns of this conversation, oldest first. Do not put the current question in `history`. Each item is:

```json
{ "role": "user", "content": "What is the leave cycle?" }
{ "role": "assistant", "content": "The leave cycle runs from 26 December to 25 December." }
```

`role` is `user` or `assistant`. `content` is a string.

Send header `X-Bypass-Cache: 1` only when you need a fresh model call. Otherwise an identical question and history is reused for 10 minutes.

### Response `200`

```json
{
  "answer": "Employees in Nashik in grades O1 and O2 get 24 privilege leave days per year.",
  "context": [
    {
      "source": "Hirect-Leave Policy.pdf",
      "page": 2,
      "section": "5. TYPES OF LEAVE",
      "type": "policy",
      "text": "Hirect Leave Policy — 5. TYPES OF LEAVE\n\n..."
    }
  ],
  "elapsed_ms": 4120.5,
  "queue_ms": 0,
  "cached": false
}
```

| Field | Meaning |
|---|---|
| `answer` | The assistant text to show in the chat |
| `context` | The passages that were given to the model. Show these as sources |
| `context[].source` | PDF file name |
| `context[].page` | Page number in that PDF, or `null` |
| `context[].section` | Heading, holiday name, or employee name |
| `context[].type` | `policy`, `holiday`, `promotion`, or `page` |
| `context[].text` | The passage, including a short headline and the original wording |
| `elapsed_ms` | Time spent producing this answer, excluding the queue |
| `queue_ms` | Time spent waiting for a free model slot |
| `cached` | `true` when this exact question was already answered recently |

### Errors

The body is `{"detail":"..."}`.

| Status | When |
|---|---|
| `400` | `question` is blank, or a history item has a bad role or non-string content |
| `429` | Too many requests are already waiting. Header `Retry-After: 5`. Retry after a few seconds |
| `500` | Retrieval or the model failed. `detail` is the error text |
| `503` | The process is still loading the model |

### React example

```javascript
export async function ask(question, history) {
  const response = await fetch("http://127.0.0.1:8000/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}
```

After a successful answer, append both turns before the next call:

```javascript
history.push(
  { role: "user", content: question },
  { role: "assistant", content: payload.answer },
);
```

## Stream the answer

`POST /api/ask/stream`

The request body is the same as `/api/ask`. The response is `text/event-stream`.

Validation and queue errors (`400`, `429`, `503`) are normal HTTP responses, before any events. A failure after the stream has started is an `error` event, and the HTTP status stays `200`.

Events are separated by a blank line. The `data` line is one JSON object.

### `context`

Sent once, before any answer text. Render these sources immediately.

```text
event: context
data: {"context":[{"source":"Hirect-Leave Policy.pdf","page":2,"section":"5. TYPES OF LEAVE","type":"policy","text":"..."}]}
```

### `token`

Sent many times, in order. Append `text` to the assistant message. The pieces are not words; they are whatever the model emitted, so concatenate them with no extra spaces.

```text
event: token
data: {"text":"Employees in Nashik"}
```

### `done`

Sent once at the end of a successful stream. Stop reading.

```text
event: done
data: {"elapsed_ms":4680.2,"queue_ms":12.4,"cached":false}
```

A cache hit sends `context`, then one `token` containing the whole answer, then `done` with `"cached": true` and `"elapsed_ms": 0`.

### `error`

Sent if generation fails after the response has started. No `done` event follows. Keep any tokens already shown and surface `detail`.

```text
event: error
data: {"detail":"connection to the model failed"}
```

### React example

```javascript
export async function askStream(question, history, { onContext, onToken }) {
  const response = await fetch("http://127.0.0.1:8000/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || response.statusText);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "context") onContext(payload.context);
      if (event === "token") {
        answer += payload.text;
        onToken(answer);
      }
      if (event === "error") throw new Error(payload.detail || "Stream failed");
      if (event === "done") return { answer, ...payload };
    }
  }

  return { answer };
}
```

`onToken` receives the full answer so far, so the chat bubble can replace its text on each event.

## Which call to use

Use `POST /api/ask` when the screen can wait and then show the finished answer and its sources together.

Use `POST /api/ask/stream` when the answer should appear word by word. Sources arrive in the `context` event before the first token, so the source panel can fill in while the answer is still being written.

Both calls share the same queue and the same answer cache. Sending the same question to one and then the other can return the stored answer.
