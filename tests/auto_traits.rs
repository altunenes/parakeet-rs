//! Sortformer is Send + Sync. A session router becomes part of it, so its trait bounds have to
//! keep both — this fails to compile if they ever stop doing so.

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn sortformer_is_send_and_sync() {
    assert_send::<parakeet_rs::sortformer::Sortformer>();
    assert_sync::<parakeet_rs::sortformer::Sortformer>();
}
