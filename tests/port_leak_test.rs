use tokio::net::TcpListener;
use std::io::Read;

#[tokio::test]
async fn test_port_not_leaked() {
    // Bind a port like spawn_health_endpoint does
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    println!("TEST1 bound port {}", port);
    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, response).await;
            }
        }
    });
    // Drop the JoinHandle — port should be released when tokio runtime drops
    println!("TEST1 port {} should be released after test ends", port);
}

#[tokio::test]
async fn test_check_port_free() {
    // Try to bind the same port — if it fails, tokio tasks leaked
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    // Just connect to verify the port is NOT already bound
    let is_bound = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok();
    println!("TEST2: randomly assigned port {} - already bound: {}", port, is_bound);
    assert!(!is_bound, "PORT LEAK DETECTED: random port {} is already occupied!", port);
    drop(listener);
}
