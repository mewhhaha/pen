fn main() {
    // Packaged builds ship Steam's library beside the executable.
    if std::env::var_os("CARGO_FEATURE_STEAM").is_some() {
        match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
            Ok("linux") => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
            Ok("macos") => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path"),
            _ => {}
        }
    }
}
