use tokio::net::{TcpListener, TcpStream};

#[tokio::test]
async fn test_spawn_endpoint() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    println!("LEAKER: bound port {}", port);
    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, response).await;
            }
        }
    });
    // Keep listener handle alive
    println!("LEAKER: letting port {} go out of scope without aborting task", port);
}

#[tokio::test]
async fn test_is_port_bound_after_leak() {
    // Try many random ports to find one that's leaked
    for _ in 0..20 {
        let is_bound = TcpStream::connect(("127.0.0.1", 0)).await.is_ok();
        if is_bound {
            println!("FOUND: random port 0 bound check returned true!");
        }
    }
    // Try to bind 10 random ports and see if any fail
    for i in 0..10 {
        match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => {
                let port = listener.local_addr().unwrap().port();
                // Immediately check if this port looks bound from another test
                println!("CONSUMER {}: bound port {}, ok", i, port);
            }
            Err(e) => {
                println!("CONSUMER {}: bind failed! {}", i, e);
            }
        }
    }
    println!("CONSUMER: done - no port conflicts found (random ports are unique per test run)");
}
