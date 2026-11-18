/// Keep the head and tail of a long string, dropping the middle.
/// Always cuts on UTF-8 character boundaries.
pub fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let half = max / 2;
    let mut head_end = half;
    while head_end > 0 && !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len() - half;
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = tail_start - head_end;
    format!(
        "{}\n... [{} bytes omitted] ...\n{}",
        &s[..head_end],
        omitted,
        &s[tail_start..]
    )
}

/// Truncate to at most `n` characters, appending an ellipsis if cut.
pub fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let mut out: String = s.chars().take(n).collect();
    out.push('…');
    out
}

/// Find the outermost JSON object in a model response (handles ```json fences and prose).
pub fn extract_json(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(&text[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_middle_short_is_identity() {
        assert_eq!(truncate_middle("hello", 10), "hello");
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        let s = "a".repeat(50) + &"b".repeat(50);
        let t = truncate_middle(&s, 20);
        assert!(t.starts_with("aaaaaaaaaa"));
        assert!(t.ends_with("bbbbbbbbbb"));
        assert!(t.contains("omitted"));
    }

    #[test]
    fn truncate_middle_is_utf8_safe() {
        let s = "é".repeat(100);
        let t = truncate_middle(&s, 15);
        assert!(t.contains("omitted"));
    }

    #[test]
    fn extract_json_from_fenced_block() {
        let text = "Sure!\n```json\n{\"a\": {\"b\": 1}}\n```\n";
        assert_eq!(extract_json(text), Some("{\"a\": {\"b\": 1}}"));
        assert_eq!(extract_json("no json"), None);
    }
}
