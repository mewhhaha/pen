use super::*;

const INK: &str = r#"{"format":"ink-persist-v1"}"#;

#[test]
fn only_an_explicit_adapter_interprets_ink_payloads() {
    let state = [5678.0, 26.0, 1.0, 11.0, 1.0, 0.0, 4.0, 9.0, 0.5, 4.0, 2.0];
    let response = [5680.0, 1.0, 11.0, 2.0, 1.0, 2.0];
    let opaque = adapter(state::OPAQUE_CONFIG).unwrap();
    assert!(opaque.sections(&state).is_none());
    assert!(opaque.tuned_values(&state).is_none());
    assert!(opaque.report(&response).is_none());
    assert!(opaque.migrate(&state, &response).unwrap().is_none());

    let ink = adapter(INK).unwrap();
    assert_eq!(ink.sections(&state).unwrap()[0].key, 11);
    assert_eq!(ink.tuned_values(&state).unwrap(), HashMap::from([(9, 0.5), (4, 2.0)]));
    assert!(ink.report(&response).unwrap().contains("reset section 11"));
    assert!(adapter(r#"{"types":{}}"#).is_err());
    assert!(adapter(r#"{"format":"unknown"}"#).is_err());
    assert!(adapter(r#"{"format":"opaque","types":{}}"#).is_err());
}

#[test]
fn malformed_views_are_unavailable_instead_of_partially_readable() {
    let ink = adapter(INK).unwrap();
    let state = [5678.0, 26.0, 2.0, 1.0, 1.0, 0.0, 1.0, 7.0, 11.0, 1.0, 0.0, 2.0, 9.0, 0.5];
    for end in 0..state.len() {
        assert!(ink.sections(&state[..end]).is_none(), "truncated at {end}");
        assert!(ink.tuned_values(&state[..end]).is_none(), "truncated at {end}");
    }
    assert!(ink.sections(&[5678.0, 26.0, 0.0, 1.0]).is_none());
    assert!(ink.sections(&[5678.0, 26.0, 1e30]).is_none());
    assert!(ink.tuned_values(&[5678.0, 26.0, 1.0, 11.0, 1.0, 0.0, 1.0, 9.0]).is_none());
    assert!(ink.tuned_values(&[5678.0, 26.0, 0.0]).unwrap().is_empty());
}

#[test]
fn a_migration_reports_what_the_second_restore_actually_did() {
    let ink = adapter(
        r#"{"format":"ink-persist-v1","types":{"score":{
        "key":100,"kind":"resource","versions":{
            "1":[{"name":"points","type":"f32"}],
            "2":[{"name":"points","type":"f32"},{"name":"bonus","type":"f32","default":0.5}]
        }}}}"#,
    )
    .unwrap();
    let saved = [5678.0, 1.0, 1.0, 100.0, 1.0, 0.0, 1.0, 42.0];
    let reset = [5680.0, 1.0, 100.0, 2.0, 1.0, 2.0];
    let migration = ink.migrate(&saved, &reset).unwrap().unwrap();
    assert_eq!(migration.state(), [5678.0, 1.0, 1.0, 100.0, 2.0, 0.0, 2.0, 42.0, 0.5]);
    let kept = [5680.0, 1.0, 100.0, 0.0, 2.0, 2.0];
    assert!(migration.report(&kept).unwrap().contains("from version 1 to 2 as saved.json declares"));
    assert!(migration.report(&reset).unwrap().contains("reset section 100"));
    assert!(migration.report(&[1.0, 2.0]).is_none());
    // The guest's own migration always takes priority over the adapter.
    assert!(ink.migrate(&saved, &[5680.0, 1.0, 100.0, 1.0, 1.0, 2.0]).unwrap().is_none());
    // Invalid payloads fail without panicking in the fallback migration.
    assert!(ink.migrate(&[], &reset).is_err());
}
