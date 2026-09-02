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
    // Bind, release, then re-bind the same port — if the second bind
    // fails, something leaked the socket after drop.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let rebind = TcpListener::bind(("127.0.0.1", port)).await;
    println!("TEST2: port {} re-bindable after drop: {}", port, rebind.is_ok());
    assert!(rebind.is_ok(), "PORT LEAK DETECTED: port {} still occupied after drop!", port);
}
