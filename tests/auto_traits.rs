//! A session router becomes part of Sortformer, so its bounds have to keep both auto traits.

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn sortformer_is_send_and_sync() {
    assert_send::<parakeet_rs::sortformer::Sortformer>();
    assert_sync::<parakeet_rs::sortformer::Sortformer>();
}
