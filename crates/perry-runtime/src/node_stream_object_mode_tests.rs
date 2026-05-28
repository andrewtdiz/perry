//! Object-mode stream behavior tests for [`super`] (`node_stream.rs`).

use super::*;
use std::cell::RefCell;

thread_local! {
    static READABLE_EVENT_READS: RefCell<Vec<(f64, f64, u64)>> = const { RefCell::new(Vec::new()) };
}

fn string_value(s: &str) -> f64 {
    let ptr = crate::string::js_string_from_bytes(s.as_ptr(), s.len() as u32);
    box_string(ptr)
}

fn object_mode_opts() -> f64 {
    let opts = crate::object::js_object_alloc(0, 1);
    js_object_set_field_by_name(opts, hidden_key(b"objectMode"), f64::from_bits(TAG_TRUE));
    box_pointer(opts as *const u8)
}

fn object_with_number(key: &[u8], value: f64) -> f64 {
    let obj = crate::object::js_object_alloc(0, 1);
    js_object_set_field_by_name(obj, hidden_key(key), value);
    box_pointer(obj as *const u8)
}

fn object_number(value: f64, key: &[u8]) -> f64 {
    let raw = raw_ptr_from_value(value);
    if raw < 0x10000 {
        return f64::from_bits(TAG_UNDEFINED);
    }
    js_object_get_field_by_name_f64(raw as *const ObjectHeader, hidden_key(key))
}

#[test]
fn object_mode_read_returns_one_stored_value_per_call() {
    let stream = js_node_stream_readable_new(object_mode_opts());
    let handle = raw_ptr_from_value(stream) as i64;
    let first = object_with_number(b"a", 1.0);
    let second = object_with_number(b"b", 2.0);

    assert_eq!(
        js_node_stream_method_push(handle, first).to_bits(),
        TAG_TRUE
    );
    assert_eq!(
        js_node_stream_method_push(handle, second).to_bits(),
        TAG_TRUE
    );
    assert_eq!(
        js_node_stream_method_push(handle, f64::from_bits(TAG_NULL)).to_bits(),
        TAG_FALSE
    );
    assert_eq!(js_node_stream_method_readable_length(handle), 2.0);

    let got_first = js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED));
    assert_eq!(got_first.to_bits(), first.to_bits());
    assert_eq!(object_number(got_first, b"a"), 1.0);
    assert_eq!(js_node_stream_method_readable_length(handle), 1.0);

    let got_second = js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED));
    assert_eq!(got_second.to_bits(), second.to_bits());
    assert_eq!(object_number(got_second, b"b"), 2.0);
    assert_eq!(js_node_stream_method_readable_length(handle), 0.0);

    assert_eq!(
        js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED)).to_bits(),
        TAG_NULL
    );
}

extern "C" fn push_two_objects_from_read(_closure: *const ClosureHeader) -> f64 {
    let stream = crate::object::js_implicit_this_get();
    let _ = push_chunk(stream, object_with_number(b"a", 1.0));
    let _ = push_chunk(stream, object_with_number(b"b", 2.0));
    let _ = push_chunk(stream, f64::from_bits(TAG_NULL));
    f64::from_bits(TAG_UNDEFINED)
}

extern "C" fn record_object_mode_reads(closure: *const ClosureHeader) -> f64 {
    let stream = crate::closure::js_closure_get_capture_f64(closure, 0);
    let handle = raw_ptr_from_value(stream) as i64;
    let first = js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED));
    let second = js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED));
    let third = js_node_stream_method_read(handle, f64::from_bits(TAG_UNDEFINED));
    READABLE_EVENT_READS.with(|reads| {
        reads.borrow_mut().push((
            object_number(first, b"a"),
            object_number(second, b"b"),
            third.to_bits(),
        ))
    });
    f64::from_bits(TAG_UNDEFINED)
}

#[test]
fn readable_listener_invokes_read_callback_before_read_event() {
    READABLE_EVENT_READS.with(|reads| reads.borrow_mut().clear());
    crate::closure::js_register_closure_arity(push_two_objects_from_read as *const u8, 0);
    crate::closure::js_register_closure_arity(record_object_mode_reads as *const u8, 0);

    let opts = crate::object::js_object_alloc(0, 2);
    js_object_set_field_by_name(opts, hidden_key(b"objectMode"), f64::from_bits(TAG_TRUE));
    let read = js_closure_alloc(push_two_objects_from_read as *const u8, 0);
    js_object_set_field_by_name(opts, hidden_key(b"read"), box_pointer(read as *const u8));
    let stream = js_node_stream_readable_new(box_pointer(opts as *const u8));
    let handle = raw_ptr_from_value(stream) as i64;

    let listener = js_closure_alloc(record_object_mode_reads as *const u8, 1);
    js_closure_set_capture_f64(listener, 0, stream);
    let _ = js_node_stream_method_on(
        handle,
        string_value("readable"),
        box_pointer(listener as *const u8),
    );
    let _ = crate::promise::js_promise_run_microtasks();

    READABLE_EVENT_READS.with(|reads| {
        assert_eq!(reads.borrow().as_slice(), &[(1.0, 2.0, TAG_NULL)]);
    });
}
