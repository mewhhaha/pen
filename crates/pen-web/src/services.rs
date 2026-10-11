use pen_protocol::{ServiceRequest, clipboard_op, pathfinding_op, platform_op, service, url_op};
use pen_runtime::services::{Answer, text};
use wasm_bindgen_futures::JsFuture;

pub async fn answer(request: &ServiceRequest) -> Answer {
    let window = web_sys::window().unwrap();
    match request.service {
        service::STEAM => Answer::unavailable(),
        service::PLATFORM => match request.operation {
            platform_op::OS => Answer::text("web"),
            platform_op::LOCALE => Answer::text(&window.navigator().language().unwrap_or_else(|| "en".into())),
            platform_op::DEVICE => Answer::text("browser"),
            _ => Answer::unknown(),
        },
        service::PATHFINDING => {
            if request.operation != pathfinding_op::FIND { return Answer::unknown(); }
            let numbers = [request.a, request.b, request.c, request.d];
            if numbers.iter().any(|n| !n.is_finite() || *n < 0.0 || n.fract() != 0.0) { return Answer::failed(); }
            let [columns, rows, start, goal] = numbers.map(|n| n as usize);
            if columns == 0 || columns.checked_mul(rows) != Some(request.values.len()) { return Answer::failed(); }
            let open: Vec<bool> = request.values.iter().map(|n| *n == 0.0).collect();
            match pen_runtime::pathfinding::find(columns, &open, start, goal) {
                Some(path) => Answer { values: path.iter().map(|n| *n as f32).collect(), ..Answer::ok(path.len() as f32) },
                None => Answer::ok(0.0),
            }
        }
        service::URL => {
            if request.operation != url_op::OPEN { return Answer::unknown(); }
            let address = text(&request.values);
            if !(address.starts_with("https://") || address.starts_with("http://")) || address.len() > 2048
                || address.chars().any(|c| c.is_whitespace() || c.is_control() || c == '"') { return Answer::failed(); }
            match window.open_with_url_and_target_and_features(&address, "_blank", "noopener,noreferrer") {
                Ok(Some(_)) => Answer::ok(1.0),
                _ => Answer::unavailable(),
            }
        }
        service::CLIPBOARD => {
            let clipboard = window.navigator().clipboard();
            match request.operation {
                clipboard_op::COPY => match JsFuture::from(clipboard.write_text(&text(&request.values))).await {
                    Ok(_) => Answer::ok(1.0), Err(_) => Answer::unavailable(),
                },
                clipboard_op::PASTE => match JsFuture::from(clipboard.read_text()).await {
                    Ok(value) => Answer::text(&value.as_string().unwrap_or_default()), Err(_) => Answer::unavailable(),
                },
                _ => Answer::unknown(),
            }
        }
        _ => Answer::unknown(),
    }
}
