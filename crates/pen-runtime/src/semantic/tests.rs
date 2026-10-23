use super::*;
use pen_protocol::service_status;

fn id(generation: u32) -> Identity {
    Identity { kind: "Hauler".into(), words: vec![7, u32::MAX, 2, generation] }
}

fn sample(tick: u64, destination: u32) -> Observation {
    Observation {
        version: 1,
        schema: Schema {
            id: "fixture-v1".into(),
            types: vec![Type {
                id: 1,
                label: "Hauler".into(),
                descriptor: Descriptor {
                    version: 1,
                    variants: vec![Variant {
                        tag: 0,
                        label: "Hauling".into(),
                        fields: vec![
                            Field {
                                id: 17,
                                label: "destination".into(),
                                type_name: "U32".into(),
                                kind: Kind::Unsigned,
                            },
                            Field { id: 31, label: "capacity".into(), type_name: "F32".into(), kind: Kind::Float },
                        ],
                    }],
                },
            }],
        },
        tick: tick.to_string(),
        objects: vec![Object {
            id: id(0),
            type_id: 1,
            tag: 0,
            fields: vec![
                FieldValue { id: 17, value: Value::Unsigned(destination) },
                FieldValue { id: 31, value: Value::Float(2.5) },
            ],
        }],
        decisions: Some(vec![]),
        omitted_objects: 0,
        omitted_decisions: 0,
    }
}

fn frame(sample: &Observation) -> RenderFrame {
    RenderFrame { extensions: vec![sample.extension(42).unwrap()], ..RenderFrame::default() }
}

fn start() -> Observations {
    let mut seen = Observations::default();
    assert_eq!(seen.answer(Request::Capture { tag: Some(42) }).status, service_status::OK);
    seen
}

fn query(seen: &mut Observations, query: Json) -> Json {
    let request = request(&code_points(&query.to_string())).unwrap();
    let answer = seen.answer(request);
    assert!(answer.values.len() <= pen_protocol::boundary::MAX_SERVICE_DATA);
    serde_json::from_str(&crate::services::text(&answer.values)).unwrap()
}

#[test]
fn named_values_reasons_and_exact_identities_follow_the_timeline() {
    let mut seen = start();
    seen.ran(1, &frame(&sample(50, 0)), false);
    let mut rejected = sample(51, 0);
    rejected.decisions = Some(vec![Decision {
        tick: "51".into(),
        object: id(0),
        system: "route".into(),
        source: Some(Source { file: "src/żuraw.blot".into(), start: 17, end: u32::MAX }),
        reason: "destination full".into(),
    }]);
    seen.ran(2, &frame(&rejected), false);
    seen.answer(Request::Watch {
        watch: Watch { object: id(0), field: Some(17), equals: Some(Value::Unsigned(u32::MAX)) },
    });
    seen.ran(3, &frame(&sample(52, u32::MAX)), false);
    assert_eq!(seen.take_break(), Some(3));
    assert_eq!(seen.take_break(), None);
    let values = query(&mut seen, json!({"op":"values","position":"3","object":id(0)}));
    assert_eq!(values["rows"][0]["field"]["label"], "destination");
    assert_eq!(values["rows"][0]["value"]["value"], u32::MAX);
    let changed = query(&mut seen, json!({"op":"diff","before":"2","after":"3"}));
    assert_eq!(changed["rows"].as_array().unwrap().len(), 1);
    assert_eq!(changed["rows"][0]["label"], "destination");
    assert!(changed["rows"][0].get("reason").is_none());
    let decisions = query(&mut seen, json!({"op":"decisions","from":"1","through":"3"}));
    assert_eq!(decisions["rows"][0]["decision"]["reason"], "destination full");
    assert_eq!(decisions["rows"][0]["position"], "2");
    assert_eq!(decisions["rows"][0]["decision"]["source"]["end"], u32::MAX);
    // Replaying one position preserves retained future observations until branching.
    seen.replayed(2, &frame(&rejected), false);
    assert_eq!(query(&mut seen, json!({"op":"objects","position":"3"}))["status"], "ok");
    seen.branch(2);
    assert_eq!(query(&mut seen, json!({"op":"objects","position":"3"}))["reason"], "not_retained");
}

#[test]
fn absence_overflow_restore_and_invalidation_do_not_invent_explanations() {
    let mut seen = start();
    seen.ran(1, &frame(&sample(1, 0)), false);
    let watch = || Watch { object: id(0), field: None, equals: None };
    seen.answer(Request::Watch { watch: watch() });
    seen.ran(2, &frame(&sample(2, 5)), true);
    assert_eq!(seen.take_break(), None, "a restore is not a simulation transition");
    let mut partial = sample(3, 7);
    partial.objects.clear();
    partial.omitted_objects = 10;
    partial.omitted_decisions = 9;
    seen.ran(3, &frame(&partial), false);
    assert_eq!(seen.take_break(), None);
    let changes = query(&mut seen, json!({"op":"diff","before":"2","after":"3"}));
    assert_eq!(changes["rows"][0]["change"], "unobserved");
    assert_eq!(query(&mut seen, json!({"op":"decisions","from":"3","through":"3"}))["omitted"], "9");
    seen.ran(4, &RenderFrame::default(), false);
    assert_eq!(
        query(&mut seen, json!({"op":"values","position":"4","object":id(0)}))["reason"],
        "instrumentation_disabled"
    );
    let mut no_events = sample(5, 9);
    no_events.decisions = None;
    seen.ran(5, &frame(&no_events), false);
    assert_eq!(query(&mut seen, json!({"op":"decisions","from":"5","through":"5"}))["reason"], "decisions_disabled");
    let mut changed_schema = sample(6, 9);
    changed_schema.schema.types[0].descriptor.version = 2;
    seen.ran(6, &frame(&changed_schema), false);
    assert!(!seen.watching());
    assert_eq!(query(&mut seen, json!({"op":"objects","position":"5"}))["reason"], "not_retained");
    assert_eq!(query(&mut seen, json!({"op":"status"}))["invalidated"], "schema_changed");
    seen.invalidate("build_changed");
    assert_eq!(query(&mut seen, json!({"op":"status"}))["retained"], 0);
}

#[test]
fn recycled_slots_cannot_inherit_watches_or_decisions() {
    let mut seen = start();
    seen.ran(1, &frame(&sample(1, 10)), false);
    let mut replaced = sample(2, 20);
    replaced.objects[0].id = id(1);
    seen.answer(Request::Watch { watch: Watch { object: id(0), field: Some(17), equals: Some(Value::Unsigned(20)) } });
    seen.ran(2, &frame(&replaced), false);
    assert_eq!(seen.take_break(), None);
    let changes = query(&mut seen, json!({"op":"diff","before":"1","after":"2"}));
    assert_eq!(changes["rows"][0]["change"], "destroyed");
    assert_eq!(changes["rows"][1]["change"], "created");
    assert_eq!(query(&mut seen, json!({"op":"values","position":"2","object":id(0)}))["reason"], "object_absent");
}

#[test]
fn malformed_observations_are_bounded_and_never_affect_rendered_output() {
    let good = frame(&sample(0, 0));
    let mut seen = start();
    for cut in 0..good.extensions[0].values.len() {
        let mut truncated = good.clone();
        truncated.extensions[0].values.truncate(cut);
        let before = truncated.clone();
        seen.ran(1, &truncated, false);
        assert_eq!(truncated, before);
        assert_eq!(query(&mut seen, json!({"op":"objects","position":"1"}))["reason"], "invalid_observation");
    }
    let mut oversized = good.clone();
    oversized.extensions[0].values.resize(MAX_PAYLOAD + 1, 0.0);
    seen.ran(1, &oversized, false);
    assert!(seen.retained_bytes() <= MAX_RETAINED_BYTES);
    let mut duplicate = good.clone();
    duplicate.extensions.push(good.extensions[0].clone());
    seen.ran(1, &duplicate, false);
    assert_eq!(query(&mut seen, json!({"op":"objects","position":"1"}))["reason"], "invalid_observation");
    let mut bad = sample(1, 0);
    bad.objects[0].fields[0].value = Value::Float(f32::NAN);
    assert!(bad.validate().is_err());
    bad = sample(1, 0);
    bad.objects[0].fields[1].id = 17;
    assert!(bad.validate().is_err());
    bad = sample(1, 0);
    bad.schema.types[0].descriptor.variants[0].fields[1].id = 17;
    assert!(bad.validate().is_err());
    assert!(request(&vec![0.0; MAX_REQUEST + 1]).is_err());
    assert!(request(&[f32::NAN]).is_err());
    assert!(position("01").is_none());
}

#[test]
fn retention_is_bounded_and_missing_history_is_explicit() {
    let mut seen = start();
    let rendered = frame(&sample(u64::MAX, u32::MAX));
    let last = MAX_FRAMES as u64 + 30;
    for at in 1..=last {
        seen.ran(at, &rendered, false);
    }
    assert!(seen.frames.len() <= MAX_FRAMES && seen.bytes <= MAX_RETAINED_BYTES);
    assert_eq!(query(&mut seen, json!({"op":"objects","position":"1"}))["reason"], "not_retained");
    assert_eq!(query(&mut seen, json!({"op":"objects","position":last.to_string()}))["tick"], u64::MAX.to_string());
    seen.replayed(1, &rendered, false);
    assert_eq!(
        query(&mut seen, json!({"op":"objects","position":"1"}))["status"],
        "ok",
        "a seek keeps the expired observation it just reconstructed"
    );
    seen.retain_from(last - 1);
    assert_eq!(seen.frames.len(), 1);
    seen.answer(Request::Capture { tag: None });
    seen.ran(151, &rendered, false);
    assert_eq!(seen.frames.len(), 0);
    assert_eq!(query(&mut seen, json!({"op":"objects","position":last.to_string()}))["reason"], "capture_disabled");
}

#[test]
fn canonical_fields_boolean_encoded_and_signed_zero() {
    let mut a = sample(1, 0);
    let fields = &mut a.schema.types[0].descriptor.variants[0].fields;
    fields.push(Field { id: 40, label: "ready".into(), type_name: "Bool".into(), kind: Kind::Boolean });
    fields.push(Field { id: 41, label: "route".into(), type_name: "Array U32".into(), kind: Kind::Encoded });
    a.objects[0].fields.extend([
        FieldValue { id: 40, value: Value::Boolean(false) },
        FieldValue { id: 41, value: Value::Encoded(vec![0, u32::MAX]) },
    ]);
    a.objects[0].fields[1].value = Value::Float(-0.0);
    let mut b = a.clone();
    b.objects[0].fields[1].value = Value::Float(0.0);
    b.objects[0].fields.reverse();
    a.validate().unwrap();
    b.validate().unwrap();
    b.schema.types[0].descriptor.variants[0].fields.reverse();
    let mut seen = start();
    seen.ran(1, &frame(&a), false);
    seen.ran(2, &frame(&b), false);
    assert_eq!(
        query(&mut seen, json!({"op":"diff","before":"1","after":"2"}))["total"],
        1,
        "descriptor reordering preserves stable field correspondence"
    );
}

#[test]
fn variant_changes_and_event_overflow_are_explicit() {
    let mut a = sample(1, 0);
    a.schema.types[0].descriptor.variants.push(Variant { tag: 7, label: "Stopped".into(), fields: vec![] });
    let mut b = a.clone();
    b.objects[0].tag = 7;
    b.objects[0].fields.clear();
    let mut seen = start();
    seen.ran(1, &frame(&a), false);
    seen.ran(2, &frame(&b), false);
    let answer = query(&mut seen, json!({"op":"diff","before":"1","after":"2"}));
    assert_eq!(answer["rows"][0]["change"], "variant");
    assert_eq!(answer["total"], 3, "a variant transition also names removed fields");
    b.decisions = Some(vec![
        Decision {
            tick: "1".into(),
            object: id(0),
            system: "route".into(),
            source: None,
            reason: "full".into()
        };
        MAX_DECISIONS + 1
    ]);
    assert!(b.validate().is_err());
    let bytes = serde_json::to_vec(&b).unwrap();
    let frame = RenderFrame {
        extensions: vec![Extension { tag: 42, values: bytes.into_iter().map(f32::from).collect() }],
        ..RenderFrame::default()
    };
    seen.ran(3, &frame, false);
    assert_eq!(query(&mut seen, json!({"op":"decisions","from":"3","through":"3"}))["reason"], "invalid_observation");
}

#[test]
fn large_diffs_page_without_losing_rows_and_payload_budget_evicts() {
    let mut a = sample(1, 0);
    a.schema.types[0].descriptor.variants[0].fields = (0..16)
        .map(|id| Field { id, label: format!("field {id}"), type_name: "Array U32".into(), kind: Kind::Encoded })
        .collect();
    a.objects[0].fields = (0..16).map(|id| FieldValue { id, value: Value::Encoded(vec![0; MAX_WORDS]) }).collect();
    let mut b = a.clone();
    for field in &mut b.objects[0].fields {
        field.value = Value::Encoded(vec![u32::MAX; MAX_WORDS]);
    }
    let mut seen = start();
    seen.ran(1, &frame(&a), false);
    seen.ran(2, &frame(&b), false);
    let mut first = 0;
    let mut count = 0;
    loop {
        let answer = query(&mut seen, json!({"op":"diff","before":"1","after":"2","first":first}));
        let rows = answer["rows"].as_array().unwrap();
        assert!(!rows.is_empty());
        count += rows.len();
        let Some(next) = answer["next"].as_u64() else { break };
        assert!(next > first);
        first = next;
    }
    assert_eq!(count, 16);
    let rendered = frame(&b);
    for at in 3..=200 {
        seen.ran(at, &rendered, false);
    }
    assert!(seen.frames.len() < MAX_FRAMES && seen.bytes <= MAX_RETAINED_BYTES);
}

#[test]
#[ignore = "opt-in release measurement; reports host capture and query cost separately"]
fn semantic_overhead() {
    use std::hint::black_box;
    use std::time::Instant;
    let rendered = frame(&sample(2, u32::MAX));
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    let iterations = 5000;
    for repetition in 0..7 {
        for offset in 0..3 {
            let mode = (offset + repetition) % 3;
            let mut seen = if mode == 0 { Observations::default() } else { start() };
            seen.ran(1, &rendered, false);
            let started = Instant::now();
            for at in 2..iterations + 2 {
                if mode == 2 {
                    black_box(seen.answer(Request::Values { position: "1".into(), object: id(0), first: 0 }));
                } else {
                    seen.ran(at, black_box(&rendered), false);
                }
            }
            samples[mode].push(started.elapsed().as_secs_f64() * 1e6 / iterations as f64);
            black_box(seen);
        }
    }
    let mut report = json!({"iterations":iterations,"payload_bytes":rendered.extensions[0].values.len()});
    for (name, mut values) in ["disabled", "capture", "query"].into_iter().zip(samples) {
        values.sort_by(f64::total_cmp);
        report[name] = json!({"median_us":values[3],"samples_us":values});
    }
    println!("{report:#}");
    if let Some(path) = std::env::var_os("PEN_SEMANTIC_FIXTURES") {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(std::path::Path::new(&path).join("timing.json"), serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
    }
}

#[test]
fn unchanged_objects_share_storage_for_a_minute_and_changed_history_stays_bounded() {
    let mut seen = start();
    for tick in 1..=MAX_FRAMES as u64 {
        seen.ran(tick, &frame(&sample(tick, 7)), false);
    }
    assert_eq!(seen.frames.len(), MAX_FRAMES);
    let first = seen.frames.front().unwrap().observation.as_ref().unwrap();
    let last = seen.frames.back().unwrap().observation.as_ref().unwrap();
    assert!(Arc::ptr_eq(&first.schema, &last.schema));
    assert!(Arc::ptr_eq(&first.objects[0], &last.objects[0]));
    assert!(seen.retained_bytes() < MAX_RETAINED_BYTES);
    println!("one minute: {} frames, {} retained bytes", seen.frames.len(), seen.retained_bytes());
    let mut large = sample(1, 7);
    large.schema.types[0].descriptor.variants[0].fields[0].kind = Kind::Encoded;
    for tick in 1..=5000 {
        large.tick = tick.to_string();
        large.objects[0].fields[0].value = Value::Encoded(vec![u32::try_from(tick).unwrap(); MAX_WORDS]);
        seen.ran(tick, &frame(&large), false);
        assert!(seen.retained_bytes() <= MAX_RETAINED_BYTES);
    }
    assert!(seen.frames.len() < MAX_FRAMES);
    // Branch/replay removal must count shared allocations only once.
    seen.branch(4990);
    seen.replayed(4990, &frame(&large), true);
    seen.retain_from(4990);
    assert_eq!(seen.frames.len(), 1);
    seen.invalidate("test");
    assert_eq!(seen.bytes, 0);
}
