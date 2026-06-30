//! Generic SSE framing for provider streaming responses.
//!
//! [`SseBuffer`] splits a raw byte stream into per-event `data:` payloads. The
//! provider-specific accumulators that turn those payloads into text deltas,
//! tool calls, and usage live in each provider module (see `providers/`).

/// Accumulates raw response bytes and yields complete SSE event payloads.
///
/// Carriage returns are stripped on input so events are always delimited by a
/// blank line (`\n\n`). Each yielded `String` is the concatenation of the
/// event's `data:` line contents (newline-joined for multi-line data).
#[derive(Default)]
pub(crate) struct SseBuffer {
    buf: Vec<u8>,
}

impl SseBuffer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend(chunk.iter().copied().filter(|&b| b != b'\r'));
    }

    /// Pop the next complete event's `data:` payload, or `None` if no complete
    /// event is buffered yet.
    pub fn next_event(&mut self) -> Option<String> {
        let pos = self.buf.windows(2).position(|w| w == b"\n\n")?;
        let frame: Vec<u8> = self.buf.drain(..pos + 2).collect();
        let text = String::from_utf8_lossy(&frame);
        let mut data = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest);
            }
        }
        Some(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_two_events() {
        let mut b = SseBuffer::new();
        b.push(b"event: x\ndata: one\n\ndata: two\n\n");
        assert_eq!(b.next_event().as_deref(), Some("one"));
        assert_eq!(b.next_event().as_deref(), Some("two"));
        assert_eq!(b.next_event(), None);
    }

    #[test]
    fn handles_crlf_and_partial_chunks() {
        let mut b = SseBuffer::new();
        b.push(b"data: hel"); // partial — no complete event yet
        assert_eq!(b.next_event(), None);
        b.push(b"lo\r\n\r\n");
        assert_eq!(b.next_event().as_deref(), Some("hello"));
    }
}
