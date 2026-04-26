use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .file("support/alloc_stableivalue.cpp")
        .cpp(true)
        .compile("iw_torch_stable");
    println!("cargo::rerun-if-changed=support/alloc_stableivalue.cpp");

    let lib_path = if std::env::var("CARGO_FEATURE_USE_TORCH_DEVEL").is_ok() {
        PathBuf::from("/workspace/pytorch/build/lib")
    } else {
        if std::env::var("VIRTUAL_ENV").is_err() {
            eprintln!("Source a virtualenv!");
            std::process::exit(1);
        }

        let normal_venv = PathBuf::from(&format!(
            "{}/lib/python3.13/site-packages/torch/lib",
            std::env::var("VIRTUAL_ENV").unwrap_or("".to_owned())
        ));
        let dev_venv = PathBuf::from(&format!(
            "{}/../build/lib/",
            std::env::var("VIRTUAL_ENV").unwrap_or("".to_owned())
        ));
        if normal_venv.is_dir() {
            normal_venv
        } else if dev_venv.is_dir() {
            dev_venv
        } else {
            eprintln!("This really doesn't look like a venv I can use.");
            std::process::exit(1);
        }
    };
    //println!("cargo::rerun-if-changed=build.rs");
    // let lib_path = ;
    //let lib_path = "/workspace/ivor/ml/pytorch_dev/pytorch/build/lib";
    // let lib_path = "/home/ivor/Documents/Code/rust/overlay_segmenter/repo/train/.venv/lib/python3.13/site-packages/torch/lib/";
    // Tell cargo to look for shared libraries in the specified directory
    let lib_path = lib_path.display();
    println!("cargo:rustc-link-search={lib_path}");
    println!("cargo:rustc-link-lib=iw_torch_stable");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    // println!("cargo:rustc-link-lib=torch");
    //
    if std::env::var("CARGO_FEATURE_USE_CUDA").is_ok() {
        println!("cargo:rustc-link-lib=torch_cuda");
    }
    println!("cargo:rustc-link-lib=torch_cpu");
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_cuda");
    //
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
    // println!("cargo:rustc-link-arg=-L{}", lib_path);
    //
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // println!("cargo:rustc-link-search=static:{}", out_path.display());
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_stable");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_consts.rs");
    std::fs::write(
        &dest_path,
        format!("pub const LIB_PATH: &str = \"{lib_path}\";\n"),
    )
    .unwrap();
}
