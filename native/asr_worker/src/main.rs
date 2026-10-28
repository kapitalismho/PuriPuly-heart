#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exit_code = puripuly_heart_asr_worker::run_cli(&args).await;
    std::process::exit(exit_code);
}
