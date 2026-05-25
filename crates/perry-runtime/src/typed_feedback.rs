//! Runtime typed-feedback sites.
//!
//! The optimizer-facing inline caches stay where they are. This module records
//! a separate, source-attributed view of what each generated dynamic boundary
//! has actually seen at runtime.

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

use crate::array::ArrayHeader;
use crate::object::ObjectHeader;
use crate::value::{
    BIGINT_TAG, INT32_TAG, JS_HANDLE_TAG, POINTER_MASK, POINTER_TAG, SHORT_STRING_TAG, STRING_TAG,
    TAG_FALSE, TAG_HOLE, TAG_MASK, TAG_NULL, TAG_TRUE, TAG_UNDEFINED,
};

const POLYMORPHIC_CAP: usize = 4;

static REGISTRY: LazyLock<Mutex<TypedFeedbackRegistry>> =
    LazyLock::new(|| Mutex::new(TypedFeedbackRegistry::default()));

#[cfg(test)]
pub(crate) static TYPED_FEEDBACK_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum TypedFeedbackSiteKind {
    PropertyGet = 0,
    PropertySet = 1,
    MethodCall = 2,
    ClosureCall = 3,
    ArrayElement = 4,
    NumericFieldWrite = 5,
    HelperReturn = 6,
}

impl TypedFeedbackSiteKind {
    fn from_raw(raw: u32) -> Self {
        match raw {
            1 => Self::PropertySet,
            2 => Self::MethodCall,
            3 => Self::ClosureCall,
            4 => Self::ArrayElement,
            5 => Self::NumericFieldWrite,
            6 => Self::HelperReturn,
            _ => Self::PropertyGet,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::PropertyGet => "property_get",
            Self::PropertySet => "property_set",
            Self::MethodCall => "method_call",
            Self::ClosureCall => "closure_call",
            Self::ArrayElement => "array_element",
            Self::NumericFieldWrite => "numeric_field_write",
            Self::HelperReturn => "helper_return",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum TypedFeedbackState {
    Uninitialized,
    Monomorphic,
    Polymorphic,
    Megamorphic,
}

impl TypedFeedbackState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::Monomorphic => "monomorphic",
            Self::Polymorphic => "polymorphic",
            Self::Megamorphic => "megamorphic",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservationSource {
    Property,
    Method,
    Closure,
    Array,
    NumericWrite,
    HelperReturn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Observation {
    source: ObservationSource,
    object_addr: usize,
    shape_addr: usize,
    key_hash: u64,
    class_id: u32,
    heap_type: u16,
    aux: u64,
    value_tag: u16,
}

impl Observation {
    fn same_feedback_key(&self, other: &Self) -> bool {
        if self.source != other.source {
            return false;
        }
        match self.source {
            ObservationSource::Property | ObservationSource::Method => {
                self.shape_addr == other.shape_addr
                    && self.key_hash == other.key_hash
                    && self.class_id == other.class_id
                    && self.heap_type == other.heap_type
                    && self.value_tag == other.value_tag
            }
            ObservationSource::NumericWrite => {
                self.shape_addr == other.shape_addr
                    && self.class_id == other.class_id
                    && self.heap_type == other.heap_type
                    && self.aux == other.aux
                    && self.value_tag == other.value_tag
            }
            ObservationSource::Closure => {
                self.aux == other.aux
                    && self.heap_type == other.heap_type
                    && self.value_tag == other.value_tag
            }
            ObservationSource::Array | ObservationSource::HelperReturn => {
                self.shape_addr == other.shape_addr
                    && self.class_id == other.class_id
                    && self.heap_type == other.heap_type
                    && self.aux == other.aux
                    && self.value_tag == other.value_tag
            }
        }
    }

    fn is_shape_keyed(&self) -> bool {
        matches!(
            self.source,
            ObservationSource::Property
                | ObservationSource::Method
                | ObservationSource::NumericWrite
        ) || (self.source == ObservationSource::HelperReturn
            && self.heap_type == crate::gc::GC_TYPE_OBJECT as u16)
    }

    fn roots_object_addr(&self) -> bool {
        false
    }

    fn roots_shape_addr(&self) -> bool {
        self.is_shape_keyed() && self.shape_addr != 0
    }

    fn affected_by_shape_change(&self, old_shape: usize, new_shape: usize, class_id: u32) -> bool {
        if !self.is_shape_keyed() {
            return false;
        }
        (old_shape != 0 && self.shape_addr == old_shape)
            || (new_shape != 0 && self.shape_addr == new_shape)
            || (old_shape == 0
                && self.shape_addr == 0
                && (class_id == 0 || self.class_id == 0 || self.class_id == class_id))
    }

    fn affected_by_representation_change(
        &self,
        obj_addr: usize,
        shape_addr: usize,
        class_id: u32,
        heap_type: u16,
    ) -> bool {
        if self.object_addr == obj_addr {
            return true;
        }
        if self.source == ObservationSource::Array {
            return heap_type != 0
                && self.heap_type == heap_type
                && (class_id == 0 || self.class_id == 0 || self.class_id == class_id);
        }
        if !self.is_shape_keyed() {
            return false;
        }
        if shape_addr != 0 {
            return self.shape_addr == shape_addr;
        }
        self.shape_addr == 0
            && (class_id == 0 || self.class_id == 0 || self.class_id == class_id)
            && (heap_type == 0 || self.heap_type == 0 || self.heap_type == heap_type)
    }
}

#[derive(Clone, Debug)]
struct SiteMetadata {
    kind: TypedFeedbackSiteKind,
    module: String,
    function: String,
    source_label: String,
    operation: String,
    guard_name: String,
    fallback_name: String,
}

#[derive(Clone, Debug)]
struct TypedFeedbackSite {
    site_id: u64,
    metadata: SiteMetadata,
    observations: Vec<Observation>,
    megamorphic: bool,
    observed_count: u64,
    guard_passes: u64,
    guard_failures: u64,
    fallback_calls: u64,
    shape_invalidations: u64,
    method_invalidations: u64,
    representation_invalidations: u64,
}

impl TypedFeedbackSite {
    fn new(site_id: u64, metadata: SiteMetadata) -> Self {
        Self {
            site_id,
            metadata,
            observations: Vec::new(),
            megamorphic: false,
            observed_count: 0,
            guard_passes: 0,
            guard_failures: 0,
            fallback_calls: 0,
            shape_invalidations: 0,
            method_invalidations: 0,
            representation_invalidations: 0,
        }
    }

    fn state(&self) -> TypedFeedbackState {
        if self.megamorphic {
            TypedFeedbackState::Megamorphic
        } else {
            match self.observations.len() {
                0 => TypedFeedbackState::Uninitialized,
                1 => TypedFeedbackState::Monomorphic,
                _ => TypedFeedbackState::Polymorphic,
            }
        }
    }

    fn observe(&mut self, observation: Observation) {
        self.observed_count = self.observed_count.saturating_add(1);
        if self.megamorphic
            || self
                .observations
                .iter()
                .any(|seen| seen.same_feedback_key(&observation))
        {
            return;
        }
        if self.observations.len() < POLYMORPHIC_CAP {
            self.observations.push(observation);
        } else {
            self.megamorphic = true;
        }
    }
}

#[derive(Default)]
struct TypedFeedbackRegistry {
    sites: HashMap<u64, TypedFeedbackSite>,
    shape_invalidations: u64,
    method_invalidations: u64,
    representation_invalidations: u64,
}

#[derive(Debug, Clone, Default)]
pub struct TypedFeedbackSnapshot {
    pub total_sites: usize,
    pub by_kind: BTreeMap<String, u64>,
    pub by_state: BTreeMap<String, u64>,
    pub shape_invalidations: u64,
    pub method_invalidations: u64,
    pub representation_invalidations: u64,
    pub guard_passes: u64,
    pub guard_failures: u64,
    pub fallback_calls: u64,
    pub guards_by_name: BTreeMap<String, GuardCounterSnapshot>,
    pub sites: Vec<TypedFeedbackSiteSnapshot>,
}

#[derive(Debug, Clone)]
pub struct GuardCounterSnapshot {
    pub passes: u64,
    pub failures: u64,
    pub fallback_calls: u64,
}

impl GuardCounterSnapshot {
    fn add_site(&mut self, site: &TypedFeedbackSite) {
        self.passes = self.passes.saturating_add(site.guard_passes);
        self.failures = self.failures.saturating_add(site.guard_failures);
        self.fallback_calls = self.fallback_calls.saturating_add(site.fallback_calls);
    }
}

#[derive(Debug, Clone)]
pub struct TypedFeedbackSiteSnapshot {
    pub site_id: u64,
    pub kind: &'static str,
    pub state: &'static str,
    pub module: String,
    pub function: String,
    pub source_label: String,
    pub operation: String,
    pub guard_name: String,
    pub fallback_name: String,
    pub observed_count: u64,
    pub observation_count: usize,
    pub guard_passes: u64,
    pub guard_failures: u64,
    pub fallback_calls: u64,
    pub shape_invalidations: u64,
    pub method_invalidations: u64,
    pub representation_invalidations: u64,
}

fn read_static_str(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 || len > 16 * 1024 {
        return String::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).unwrap_or("").to_string()
}

fn registry() -> crate::gc::GcRootRegistryGuard<'static, TypedFeedbackRegistry> {
    crate::gc::lock_gc_root_registry(&REGISTRY)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_register_site(
    site_id: u64,
    kind: u32,
    module_ptr: *const u8,
    module_len: usize,
    function_ptr: *const u8,
    function_len: usize,
    source_ptr: *const u8,
    source_len: usize,
    operation_ptr: *const u8,
    operation_len: usize,
    guard_ptr: *const u8,
    guard_len: usize,
    fallback_ptr: *const u8,
    fallback_len: usize,
) {
    if site_id == 0 {
        return;
    }
    let metadata = SiteMetadata {
        kind: TypedFeedbackSiteKind::from_raw(kind),
        module: read_static_str(module_ptr, module_len),
        function: read_static_str(function_ptr, function_len),
        source_label: read_static_str(source_ptr, source_len),
        operation: read_static_str(operation_ptr, operation_len),
        guard_name: read_static_str(guard_ptr, guard_len),
        fallback_name: read_static_str(fallback_ptr, fallback_len),
    };
    let mut reg = registry();
    reg.sites
        .entry(site_id)
        .and_modify(|site| site.metadata = metadata.clone())
        .or_insert_with(|| TypedFeedbackSite::new(site_id, metadata));
}

fn value_tag(bits: u64) -> u16 {
    (bits >> 48) as u16
}

fn value_pointer(bits: u64) -> usize {
    let tag = bits & TAG_MASK;
    if tag == POINTER_TAG || tag == STRING_TAG || tag == BIGINT_TAG {
        (bits & POINTER_MASK) as usize
    } else {
        0
    }
}

const STABLE_VALUE_NUMBER: u16 = 1;
const STABLE_VALUE_BOOLEAN: u16 = 2;
const STABLE_VALUE_NULL: u16 = 3;
const STABLE_VALUE_UNDEFINED: u16 = 4;
const STABLE_VALUE_HOLE: u16 = 5;
const STABLE_VALUE_SHORT_STRING: u16 = 6;
const STABLE_VALUE_STRING: u16 = 7;
const STABLE_VALUE_BIGINT: u16 = 8;
const STABLE_VALUE_POINTER: u16 = 9;
const STABLE_VALUE_INT32: u16 = 10;
const STABLE_VALUE_JS_HANDLE: u16 = 11;

const ARRAY_ACCESS_UNKNOWN: u8 = 0;
const ARRAY_ACCESS_INDEXED_IN_BOUNDS: u8 = 1;
const ARRAY_ACCESS_INDEXED_OUT_OF_BOUNDS: u8 = 2;
const ARRAY_ACCESS_STRING_KEY: u8 = 3;

const ARRAY_LAYOUT_INVALID: u8 = 0;
const ARRAY_LAYOUT_EMPTY: u8 = 1;
const ARRAY_LAYOUT_POINTER_FREE: u8 = 2;
const ARRAY_LAYOUT_POINTER_ONLY: u8 = 3;
const ARRAY_LAYOUT_MIXED: u8 = 4;
const ARRAY_LAYOUT_UNKNOWN: u8 = 5;
const ARRAY_LAYOUT_BUFFER: u8 = 6;
const ARRAY_LAYOUT_TYPED_ARRAY: u8 = 7;
const ARRAY_LAYOUT_LAZY: u8 = 8;

fn stable_value_kind(bits: u64) -> u16 {
    match bits {
        TAG_TRUE | TAG_FALSE => return STABLE_VALUE_BOOLEAN,
        TAG_NULL => return STABLE_VALUE_NULL,
        TAG_UNDEFINED => return STABLE_VALUE_UNDEFINED,
        TAG_HOLE => return STABLE_VALUE_HOLE,
        _ => {}
    }

    match bits & TAG_MASK {
        POINTER_TAG => STABLE_VALUE_POINTER,
        STRING_TAG => STABLE_VALUE_STRING,
        BIGINT_TAG => STABLE_VALUE_BIGINT,
        JS_HANDLE_TAG => STABLE_VALUE_JS_HANDLE,
        SHORT_STRING_TAG => STABLE_VALUE_SHORT_STRING,
        INT32_TAG => STABLE_VALUE_INT32,
        _ => STABLE_VALUE_NUMBER,
    }
}

fn raw_heap_type(addr: usize) -> u16 {
    if addr == 0 {
        return 0;
    }
    if crate::buffer::is_registered_buffer(addr) {
        return crate::gc::GC_TYPE_BUFFER as u16;
    }
    if crate::typedarray::lookup_typed_array_kind(addr).is_some() {
        return crate::gc::GC_TYPE_TYPED_ARRAY as u16;
    }
    if !crate::object::is_valid_obj_ptr(addr as *const u8) {
        return 0;
    }
    unsafe {
        let gc = (addr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader;
        let gc_type = (*gc).obj_type;
        if crate::gc::gc_type_info(gc_type).is_some() {
            gc_type as u16
        } else {
            0
        }
    }
}

fn pack_array_aux(access_kind: u8, layout_kind: u8, element_kind: u16, typed_kind: u8) -> u64 {
    (access_kind as u64)
        | ((layout_kind as u64) << 8)
        | ((element_kind as u64) << 16)
        | ((typed_kind as u64) << 32)
}

fn array_layout_kind(addr: usize, len: u64) -> u8 {
    if len == 0 {
        return ARRAY_LAYOUT_EMPTY;
    }

    let mut pointer_slots = 0usize;
    if crate::gc::layout_visit_pointer_slots_for_user(addr, len as usize, |_| {
        pointer_slots = pointer_slots.saturating_add(1);
    }) {
        if pointer_slots == 0 {
            ARRAY_LAYOUT_POINTER_FREE
        } else if pointer_slots as u64 == len {
            ARRAY_LAYOUT_POINTER_ONLY
        } else {
            ARRAY_LAYOUT_MIXED
        }
    } else {
        ARRAY_LAYOUT_UNKNOWN
    }
}

fn array_element_kind(addr: usize, index: Option<u32>, len: u64, layout_kind: u8) -> u16 {
    let Some(index) = index else {
        return layout_kind as u16;
    };
    if index as u64 >= len || len > 16_000_000 {
        return STABLE_VALUE_UNDEFINED;
    }
    unsafe {
        let elements = (addr as *const u8).add(std::mem::size_of::<ArrayHeader>()) as *const u64;
        stable_value_kind(*elements.add(index as usize))
    }
}

fn classify_array(addr: usize, index: Option<u32>) -> (u32, u16, u64, u16) {
    if addr == 0 {
        return (
            0,
            0,
            pack_array_aux(
                ARRAY_ACCESS_UNKNOWN,
                ARRAY_LAYOUT_INVALID,
                STABLE_VALUE_UNDEFINED,
                0,
            ),
            STABLE_VALUE_UNDEFINED,
        );
    }

    let access_kind = match index {
        None => ARRAY_ACCESS_UNKNOWN,
        Some(u32::MAX) => ARRAY_ACCESS_STRING_KEY,
        Some(_) => ARRAY_ACCESS_INDEXED_IN_BOUNDS,
    };

    if crate::buffer::is_registered_buffer(addr) {
        let len = unsafe { (*(addr as *const crate::buffer::BufferHeader)).length as u64 };
        let access_kind = match index {
            Some(i) if i != u32::MAX && i as u64 >= len => ARRAY_ACCESS_INDEXED_OUT_OF_BOUNDS,
            _ => access_kind,
        };
        let element_kind = STABLE_VALUE_NUMBER;
        return (
            crate::buffer::BUFFER_TYPE_ID,
            crate::gc::GC_TYPE_BUFFER as u16,
            pack_array_aux(access_kind, ARRAY_LAYOUT_BUFFER, element_kind, 0),
            element_kind,
        );
    }

    if let Some(kind) = crate::typedarray::lookup_typed_array_kind(addr) {
        let len = unsafe { (*(addr as *const crate::typedarray::TypedArrayHeader)).length as u64 };
        let access_kind = match index {
            Some(i) if i != u32::MAX && i as u64 >= len => ARRAY_ACCESS_INDEXED_OUT_OF_BOUNDS,
            _ => access_kind,
        };
        let element_kind = STABLE_VALUE_NUMBER;
        return (
            crate::typedarray::class_id_for_kind(kind),
            crate::gc::GC_TYPE_TYPED_ARRAY as u16,
            pack_array_aux(access_kind, ARRAY_LAYOUT_TYPED_ARRAY, element_kind, kind),
            element_kind,
        );
    }

    if !crate::object::is_valid_obj_ptr(addr as *const u8) {
        return (
            0,
            0,
            pack_array_aux(
                ARRAY_ACCESS_UNKNOWN,
                ARRAY_LAYOUT_INVALID,
                STABLE_VALUE_UNDEFINED,
                0,
            ),
            STABLE_VALUE_UNDEFINED,
        );
    }

    unsafe {
        let gc = (addr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader;
        let gc_type = (*gc).obj_type;
        if gc_type == crate::gc::GC_TYPE_LAZY_ARRAY {
            return (
                0,
                gc_type as u16,
                pack_array_aux(access_kind, ARRAY_LAYOUT_LAZY, STABLE_VALUE_POINTER, 0),
                STABLE_VALUE_POINTER,
            );
        }
        if (*gc).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0 {
            return (
                0,
                gc_type as u16,
                pack_array_aux(access_kind, ARRAY_LAYOUT_INVALID, STABLE_VALUE_POINTER, 0),
                STABLE_VALUE_POINTER,
            );
        }
        if gc_type != crate::gc::GC_TYPE_ARRAY {
            return (
                0,
                gc_type as u16,
                pack_array_aux(
                    ARRAY_ACCESS_UNKNOWN,
                    ARRAY_LAYOUT_INVALID,
                    STABLE_VALUE_POINTER,
                    0,
                ),
                STABLE_VALUE_POINTER,
            );
        }

        let len = (*(addr as *const ArrayHeader)).length as u64;
        let access_kind = match index {
            Some(i) if i != u32::MAX && i as u64 >= len => ARRAY_ACCESS_INDEXED_OUT_OF_BOUNDS,
            _ => access_kind,
        };
        let layout_kind = array_layout_kind(addr, len);
        let element_kind =
            array_element_kind(addr, index.filter(|i| *i != u32::MAX), len, layout_kind);
        (
            0,
            gc_type as u16,
            pack_array_aux(access_kind, layout_kind, element_kind, 0),
            element_kind,
        )
    }
}

fn helper_return_facts(bits: u64) -> (usize, u32, u16, u64, u16) {
    let value_kind = stable_value_kind(bits);
    let addr = value_pointer(bits);
    if addr == 0 {
        return (0, 0, 0, 0, value_kind);
    }

    if crate::buffer::is_registered_buffer(addr)
        || crate::typedarray::lookup_typed_array_kind(addr).is_some()
    {
        let (class_id, heap_type, aux, element_kind) = classify_array(addr, None);
        return (0, class_id, heap_type, aux, element_kind);
    }

    match raw_heap_type(addr) as u8 {
        crate::gc::GC_TYPE_ARRAY | crate::gc::GC_TYPE_LAZY_ARRAY => {
            let (class_id, heap_type, aux, element_kind) = classify_array(addr, None);
            (0, class_id, heap_type, aux, element_kind)
        }
        crate::gc::GC_TYPE_OBJECT => {
            let (shape_addr, class_id, heap_type) = object_shape(addr);
            (shape_addr, class_id, heap_type, 0, value_kind)
        }
        heap_type => (0, 0, heap_type as u16, 0, value_kind),
    }
}

fn normalize_raw_object_addr(bits: u64) -> usize {
    let top = bits >> 48;
    let addr = if top >= 0x7FF8 {
        bits & POINTER_MASK
    } else {
        bits
    } as usize;
    if addr < 0x10000 || (addr as u64) >> 48 != 0 {
        0
    } else {
        addr
    }
}

fn object_shape(addr: usize) -> (usize, u32, u16) {
    if addr == 0 {
        return (0, 0, 0);
    }
    let ptr = addr as *const ObjectHeader;
    if !crate::object::is_valid_obj_ptr(ptr as *const u8) {
        return (0, 0, 0);
    }
    unsafe {
        let gc = (ptr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader;
        let gc_type = (*gc).obj_type as u16;
        if (*gc).obj_type != crate::gc::GC_TYPE_OBJECT {
            return (0, 0, gc_type);
        }
        let class_id = (*ptr).class_id;
        let shape = (*ptr).keys_array as usize;
        (shape, class_id, gc_type)
    }
}

fn observe(site_id: u64, fallback_kind: TypedFeedbackSiteKind, observation: Observation) {
    if site_id == 0 {
        return;
    }
    let mut reg = registry();
    let site = reg.sites.entry(site_id).or_insert_with(|| {
        TypedFeedbackSite::new(
            site_id,
            SiteMetadata {
                kind: fallback_kind,
                module: String::new(),
                function: String::new(),
                source_label: String::new(),
                operation: String::new(),
                guard_name: String::new(),
                fallback_name: String::new(),
            },
        )
    });
    site.observe(observation);
}

fn site_entry(
    reg: &mut TypedFeedbackRegistry,
    site_id: u64,
    fallback_kind: TypedFeedbackSiteKind,
) -> &mut TypedFeedbackSite {
    reg.sites.entry(site_id).or_insert_with(|| {
        TypedFeedbackSite::new(
            site_id,
            SiteMetadata {
                kind: fallback_kind,
                module: String::new(),
                function: String::new(),
                source_label: String::new(),
                operation: String::new(),
                guard_name: String::new(),
                fallback_name: String::new(),
            },
        )
    })
}

fn guard_observe(
    site_id: u64,
    fallback_kind: TypedFeedbackSiteKind,
    observation: Observation,
    contract_valid: bool,
) -> bool {
    if site_id == 0 {
        return contract_valid;
    }
    let mut reg = registry();
    let site = site_entry(&mut reg, site_id, fallback_kind);
    let guard_passed = contract_valid
        && !site.megamorphic
        && (site.observations.is_empty()
            || site
                .observations
                .iter()
                .any(|seen| seen.same_feedback_key(&observation)));
    if guard_passed {
        site.guard_passes = site.guard_passes.saturating_add(1);
    } else {
        site.guard_failures = site.guard_failures.saturating_add(1);
    }
    site.observe(observation);
    guard_passed
}

fn record_guard_pass(site_id: u64) {
    if site_id == 0 {
        return;
    }
    let mut reg = registry();
    if let Some(site) = reg.sites.get_mut(&site_id) {
        site.guard_passes = site.guard_passes.saturating_add(1);
    }
}

fn record_guard_fail(site_id: u64) {
    if site_id == 0 {
        return;
    }
    let mut reg = registry();
    if let Some(site) = reg.sites.get_mut(&site_id) {
        site.guard_failures = site.guard_failures.saturating_add(1);
    }
}

fn record_fallback_call(site_id: u64) {
    if site_id == 0 {
        return;
    }
    let mut reg = registry();
    if let Some(site) = reg.sites.get_mut(&site_id) {
        site.fallback_calls = site.fallback_calls.saturating_add(1);
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_record_guard_pass(site_id: u64) {
    record_guard_pass(site_id);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_record_guard_fail(site_id: u64) {
    record_guard_fail(site_id);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_record_fallback_call(site_id: u64) {
    record_fallback_call(site_id);
}

fn observe_property(
    site_id: u64,
    kind: TypedFeedbackSiteKind,
    obj_bits: u64,
    key: *const crate::StringHeader,
) {
    let object_addr = normalize_raw_object_addr(obj_bits);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    observe(
        site_id,
        kind,
        Observation {
            source: ObservationSource::Property,
            object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
            shape_addr,
            key_hash: key_hash(key),
            class_id,
            heap_type: gc_type,
            aux: 0,
            value_tag: value_tag(obj_bits),
        },
    );
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_observe_property_get(
    site_id: u64,
    obj: *const ObjectHeader,
    key: *const crate::StringHeader,
) {
    observe_property(site_id, TypedFeedbackSiteKind::PropertyGet, obj as u64, key);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_observe_property_set(
    site_id: u64,
    obj: *mut ObjectHeader,
    key: *const crate::StringHeader,
) {
    observe_property(site_id, TypedFeedbackSiteKind::PropertySet, obj as u64, key);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_object_get_field_by_name_f64(
    site_id: u64,
    obj: *const ObjectHeader,
    key: *const crate::StringHeader,
) -> f64 {
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let observation = Observation {
        source: ObservationSource::Property,
        object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: 0,
        value_tag: value_tag(obj as u64),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::PropertyGet,
        observation,
        valid_string_key(key) && gc_type == crate::gc::GC_TYPE_OBJECT as u16,
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::object::js_object_get_field_by_name_f64(obj, key)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_object_set_field_by_name(
    site_id: u64,
    obj: *mut ObjectHeader,
    key: *const crate::StringHeader,
    value: f64,
) {
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let observation = Observation {
        source: ObservationSource::Property,
        object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: 0,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::PropertySet,
        observation,
        valid_string_key(key) && gc_type == crate::gc::GC_TYPE_OBJECT as u16,
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::object::js_object_set_field_by_name(obj, key, value);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_object_set_field_by_name_fast(
    site_id: u64,
    obj: *mut ObjectHeader,
    key: *const crate::StringHeader,
    value: f64,
) {
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let handled = crate::object::js_object_set_field_by_name_transition_fast(obj, key, value) != 0;
    let observation = Observation {
        source: ObservationSource::Property,
        object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: 0,
        value_tag: stable_value_kind(value.to_bits()),
    };
    guard_observe(
        site_id,
        TypedFeedbackSiteKind::PropertySet,
        observation,
        handled,
    );
    if !handled {
        record_fallback_call(site_id);
        crate::object::js_object_set_field_by_name(obj, key, value);
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn key_hash(key: *const crate::StringHeader) -> u64 {
    if key.is_null() || (key as usize) < 0x1000 {
        return 0;
    }
    unsafe {
        let len = (*key).byte_len as usize;
        if len > 4096 {
            return 0;
        }
        let data = (key as *const u8).add(std::mem::size_of::<crate::StringHeader>());
        hash_bytes(std::slice::from_raw_parts(data, len))
    }
}

fn valid_string_key(key: *const crate::StringHeader) -> bool {
    if key.is_null() || (key as usize) < 0x1000 {
        return false;
    }
    unsafe {
        let len = (*key).byte_len as usize;
        len <= 4096
    }
}

fn valid_method_name(method_name_ptr: *const i8, method_name_len: usize) -> bool {
    !method_name_ptr.is_null() && method_name_len > 0 && method_name_len <= 4096
}

fn method_name_bytes<'a>(method_name_ptr: *const i8, method_name_len: usize) -> Option<&'a [u8]> {
    if !valid_method_name(method_name_ptr, method_name_len) {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(method_name_ptr as *const u8, method_name_len) })
}

fn method_name_str<'a>(method_name_ptr: *const i8, method_name_len: usize) -> Option<&'a str> {
    std::str::from_utf8(method_name_bytes(method_name_ptr, method_name_len)?).ok()
}

fn is_plain_number_bits(bits: u64) -> bool {
    stable_value_kind(bits) == STABLE_VALUE_NUMBER
}

fn is_numeric_value_bits(bits: u64) -> bool {
    matches!(
        stable_value_kind(bits),
        STABLE_VALUE_NUMBER | STABLE_VALUE_INT32
    )
}

fn gc_header_for_user_addr(addr: usize) -> Option<*const crate::gc::GcHeader> {
    if addr < crate::gc::GC_HEADER_SIZE + 0x1000
        || (addr as u64) >> 48 != 0
        || !crate::object::is_valid_obj_ptr(addr as *const u8)
    {
        return None;
    }
    Some(unsafe {
        (addr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader
    })
}

fn plain_array_index_guard(arr: *const ArrayHeader, index: u32, require_in_bounds: bool) -> bool {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let Some(header) = gc_header_for_user_addr(raw_addr) else {
        return false;
    };
    unsafe {
        if (*header).obj_type != crate::gc::GC_TYPE_ARRAY
            || (*header).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0
        {
            return false;
        }
        let arr = raw_addr as *const ArrayHeader;
        let len = (*arr).length;
        let cap = (*arr).capacity;
        if len > 16_000_000 || cap > 16_000_000 || len > cap {
            return false;
        }
        !require_in_bounds || index < len
    }
}

fn numeric_array_index_guard(arr: *const ArrayHeader, index: u32, require_in_bounds: bool) -> bool {
    plain_array_index_guard(arr, index, require_in_bounds)
        && crate::array::js_array_is_numeric_f64_layout(arr) != 0
}

fn numeric_array_push_guard(arr: *const ArrayHeader, value: f64) -> bool {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let Some(header) = gc_header_for_user_addr(raw_addr) else {
        return false;
    };
    unsafe {
        if (*header).obj_type != crate::gc::GC_TYPE_ARRAY
            || (*header).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0
        {
            return false;
        }
        let arr = raw_addr as *const ArrayHeader;
        let len = (*arr).length;
        let cap = (*arr).capacity;
        len <= 16_000_000
            && cap <= 16_000_000
            && len < cap
            && is_numeric_value_bits(value.to_bits())
            && crate::array::js_array_is_numeric_f64_layout(arr) != 0
    }
}

fn object_key_matches_field(
    obj: *mut ObjectHeader,
    key: *const crate::StringHeader,
    field_index: u32,
) -> bool {
    if !valid_string_key(key) {
        return false;
    }
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, _, heap_type) = object_shape(object_addr);
    if heap_type != crate::gc::GC_TYPE_OBJECT as u16 || shape_addr == 0 {
        return false;
    }
    unsafe {
        let obj = object_addr as *mut ObjectHeader;
        let alloc_limit = std::cmp::max((*obj).field_count, 8);
        if field_index >= alloc_limit {
            return false;
        }
        let keys = (*obj).keys_array;
        if keys.is_null() || (keys as usize) != shape_addr {
            return false;
        }
        if !plain_array_index_guard(keys, field_index, true) {
            return false;
        }
        let stored = crate::array::js_array_get(keys, field_index);
        stored.is_string()
            && !stored.as_string_ptr().is_null()
            && crate::string::js_string_equals(key, stored.as_string_ptr()) != 0
    }
}

fn object_has_own_key_bytes(obj: *const ObjectHeader, key_bytes: &[u8]) -> bool {
    if obj.is_null() || key_bytes.is_empty() || key_bytes.len() > 4096 {
        return false;
    }
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, _, heap_type) = object_shape(object_addr);
    if heap_type != crate::gc::GC_TYPE_OBJECT as u16 || shape_addr == 0 {
        return false;
    }
    unsafe {
        let obj = object_addr as *const ObjectHeader;
        let keys = (*obj).keys_array;
        if keys.is_null() || keys as usize != shape_addr {
            return false;
        }
        let key_count = crate::array::js_array_length(keys) as usize;
        if key_count > 65_536 {
            return true;
        }
        for i in 0..key_count {
            let stored = crate::array::js_array_get(keys, i as u32);
            if !stored.is_string() {
                continue;
            }
            let string = stored.as_string_ptr();
            if string.is_null() {
                continue;
            }
            let len = (*string).byte_len as usize;
            if len != key_bytes.len() {
                continue;
            }
            let data = (string as *const u8).add(std::mem::size_of::<crate::StringHeader>());
            if std::slice::from_raw_parts(data, len) == key_bytes {
                return true;
            }
        }
        false
    }
}

fn vtable_method_matches(class_id: u32, method_name: &str, expected_func_ptr: usize) -> bool {
    if class_id == 0 || expected_func_ptr == 0 {
        return false;
    }
    let Ok(registry) = crate::object::CLASS_VTABLE_REGISTRY.read() else {
        return false;
    };
    let Some(registry) = registry.as_ref() else {
        return false;
    };
    let mut cid = class_id;
    for _ in 0..32 {
        if let Some(vtable) = registry.get(&cid) {
            if let Some(entry) = vtable.methods.get(method_name) {
                return entry.func_ptr == expected_func_ptr;
            }
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn prototype_may_override_method(class_id: u32, method_name: &str, method_bytes: &[u8]) -> bool {
    if class_id == 0 {
        return false;
    }
    if crate::object::lookup_prototype_method(class_id, method_name).is_some() {
        return true;
    }
    let mut cid = class_id;
    for _ in 0..32 {
        let proto = crate::object::class_prototype_object(cid);
        if !proto.is_null() && object_has_own_key_bytes(proto, method_bytes) {
            return true;
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn method_direct_call_contract(
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    method_name_ptr: *const i8,
    method_name_len: usize,
    expected_func_ptr: *const u8,
) -> (usize, u32, u16, u64, bool) {
    let object_addr = normalize_raw_object_addr(receiver.to_bits());
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let Some(method_bytes) = method_name_bytes(method_name_ptr, method_name_len) else {
        return (shape_addr, class_id, gc_type, 0, false);
    };
    let Some(method_name) = method_name_str(method_name_ptr, method_name_len) else {
        return (
            shape_addr,
            class_id,
            gc_type,
            hash_bytes(method_bytes),
            false,
        );
    };
    let name_hash = hash_bytes(method_bytes);
    if object_addr == 0
        || expected_class_id == 0
        || expected_keys.is_null()
        || expected_func_ptr.is_null()
    {
        return (shape_addr, class_id, gc_type, name_hash, false);
    }
    let Some(gc_header) = gc_header_for_user_addr(object_addr) else {
        return (shape_addr, class_id, gc_type, name_hash, false);
    };
    unsafe {
        if (*gc_header).obj_type != crate::gc::GC_TYPE_OBJECT
            || (*gc_header).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0
        {
            return (shape_addr, class_id, gc_type, name_hash, false);
        }
        let obj = object_addr as *const ObjectHeader;
        if (*obj).object_type != crate::error::OBJECT_TYPE_REGULAR {
            return (shape_addr, class_id, gc_type, name_hash, false);
        }
        if (*obj).class_id == crate::object::NATIVE_MODULE_CLASS_ID
            || (*obj).class_id != expected_class_id
            || (*obj).keys_array as usize != expected_keys as usize
            || shape_addr != expected_keys as usize
        {
            return (shape_addr, class_id, gc_type, name_hash, false);
        }
        if object_has_own_key_bytes(obj, method_bytes) {
            return (shape_addr, class_id, gc_type, name_hash, false);
        }
    }

    let expected_func = expected_func_ptr as usize;
    let valid = vtable_method_matches(class_id, method_name, expected_func)
        && !prototype_may_override_method(class_id, method_name, method_bytes);
    (shape_addr, class_id, gc_type, name_hash, valid)
}

fn key_as_str(key: *const crate::StringHeader) -> Option<String> {
    if !valid_string_key(key) {
        return None;
    }
    unsafe {
        let len = (*key).byte_len as usize;
        let data = (key as *const u8).add(std::mem::size_of::<crate::StringHeader>());
        std::str::from_utf8(std::slice::from_raw_parts(data, len))
            .ok()
            .map(|s| s.to_string())
    }
}

fn class_setter_in_chain(class_id: u32, key_name: &str) -> bool {
    if class_id == 0 {
        return false;
    }
    let Ok(registry) = crate::object::CLASS_VTABLE_REGISTRY.read() else {
        return true;
    };
    let Some(registry) = registry.as_ref() else {
        return false;
    };
    let mut cid = class_id;
    for _ in 0..32 {
        if registry
            .get(&cid)
            .map(|vtable| vtable.setters.contains_key(key_name))
            .unwrap_or(false)
        {
            return true;
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn class_getter_in_chain(class_id: u32, key_name: &str) -> bool {
    if class_id == 0 {
        return false;
    }
    let Ok(registry) = crate::object::CLASS_VTABLE_REGISTRY.read() else {
        return true;
    };
    let Some(registry) = registry.as_ref() else {
        return false;
    };
    let mut cid = class_id;
    for _ in 0..32 {
        if registry
            .get(&cid)
            .map(|vtable| vtable.getters.contains_key(key_name))
            .unwrap_or(false)
        {
            return true;
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn descriptor_blocks_class_field_get(obj_addr: usize, class_id: u32, key_name: &str) -> bool {
    if !crate::object::descriptors_in_use() {
        return false;
    }
    if crate::object::get_accessor_descriptor(obj_addr, key_name).is_some() {
        return true;
    }

    let mut cid = class_id;
    for _ in 0..32 {
        let proto = crate::object::class_prototype_object(cid);
        if !proto.is_null()
            && crate::object::get_accessor_descriptor(proto as usize, key_name).is_some()
        {
            return true;
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn class_field_get_contract(
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    key: *const crate::StringHeader,
    expected_field_index: u32,
    require_raw_f64: bool,
) -> (usize, u32, u16, bool) {
    let object_addr = normalize_raw_object_addr(receiver.to_bits());
    if object_addr == 0 || expected_class_id == 0 || expected_keys.is_null() {
        return (0, 0, 0, false);
    }
    let Some(gc_header) = gc_header_for_user_addr(object_addr) else {
        return (0, 0, 0, false);
    };
    unsafe {
        let gc_type = (*gc_header).obj_type as u16;
        if (*gc_header).obj_type != crate::gc::GC_TYPE_OBJECT {
            return (0, 0, gc_type, false);
        }
        if (*gc_header).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0 {
            return (0, 0, gc_type, false);
        }

        let obj = object_addr as *mut ObjectHeader;
        let class_id = (*obj).class_id;
        let shape_addr = (*obj).keys_array as usize;
        let key_name = match key_as_str(key) {
            Some(name) => name,
            None => return (shape_addr, class_id, gc_type, false),
        };
        let expected_shape_addr = expected_keys as usize;
        let valid = (*obj).object_type == crate::error::OBJECT_TYPE_REGULAR
            && class_id == expected_class_id
            && shape_addr == expected_shape_addr
            && expected_field_index < (*obj).field_count
            && plain_array_index_guard(expected_keys, expected_field_index, true)
            && object_key_matches_field(obj, key, expected_field_index)
            && (!require_raw_f64
                || crate::gc::layout_typed_raw_f64_slot_for_user(
                    object_addr,
                    expected_field_index as usize,
                ))
            && !class_getter_in_chain(class_id, &key_name)
            && !descriptor_blocks_class_field_get(object_addr, class_id, &key_name);
        (shape_addr, class_id, gc_type, valid)
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_class_field_get_guard(
    site_id: u64,
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    key: *const crate::StringHeader,
    expected_field_index: u32,
    require_raw_f64: i32,
) -> i32 {
    let (shape_addr, class_id, gc_type, contract_valid) = class_field_get_contract(
        receiver,
        expected_class_id,
        expected_keys,
        key,
        expected_field_index,
        require_raw_f64 != 0,
    );
    let object_addr = normalize_raw_object_addr(receiver.to_bits());
    let observation = Observation {
        source: ObservationSource::Property,
        object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: expected_field_index as u64,
        value_tag: value_tag(receiver.to_bits()),
    };
    if guard_observe(
        site_id,
        TypedFeedbackSiteKind::PropertyGet,
        observation,
        contract_valid,
    ) {
        1
    } else {
        0
    }
}

fn descriptor_blocks_class_field_set(obj_addr: usize, class_id: u32, key_name: &str) -> bool {
    if !crate::object::descriptors_in_use() {
        return false;
    }
    if crate::object::get_accessor_descriptor(obj_addr, key_name).is_some() {
        return true;
    }
    if crate::object::get_property_attrs(obj_addr, key_name)
        .map(|attrs| !attrs.writable())
        .unwrap_or(false)
    {
        return true;
    }

    let mut cid = class_id;
    for _ in 0..32 {
        let proto = crate::object::class_prototype_object(cid);
        if !proto.is_null() {
            let proto_addr = proto as usize;
            if crate::object::get_accessor_descriptor(proto_addr, key_name).is_some() {
                return true;
            }
            if crate::object::get_property_attrs(proto_addr, key_name)
                .map(|attrs| !attrs.writable())
                .unwrap_or(false)
            {
                return true;
            }
        }
        match crate::object::get_parent_class_id(cid) {
            Some(parent) if parent != 0 && parent != cid => cid = parent,
            _ => break,
        }
    }
    false
}

fn class_field_set_contract(
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    key: *const crate::StringHeader,
    expected_field_index: u32,
    require_raw_f64: bool,
    value_bits: u64,
) -> (usize, u32, u16, bool) {
    let object_addr = normalize_raw_object_addr(receiver.to_bits());
    if object_addr == 0 || expected_class_id == 0 || expected_keys.is_null() {
        return (0, 0, 0, false);
    }
    let Some(gc_header) = gc_header_for_user_addr(object_addr) else {
        return (0, 0, 0, false);
    };
    unsafe {
        let gc_type = (*gc_header).obj_type as u16;
        if (*gc_header).obj_type != crate::gc::GC_TYPE_OBJECT {
            return (0, 0, gc_type, false);
        }
        if (*gc_header).gc_flags & crate::gc::GC_FLAG_FORWARDED != 0 {
            return (0, 0, gc_type, false);
        }
        if (*gc_header)._reserved & crate::gc::OBJ_FLAG_FROZEN != 0 {
            let obj = object_addr as *mut ObjectHeader;
            return ((*obj).keys_array as usize, (*obj).class_id, gc_type, false);
        }

        let obj = object_addr as *mut ObjectHeader;
        let class_id = (*obj).class_id;
        let shape_addr = (*obj).keys_array as usize;
        let key_name = match key_as_str(key) {
            Some(name) => name,
            None => return (shape_addr, class_id, gc_type, false),
        };
        let expected_shape_addr = expected_keys as usize;
        let valid = class_id == expected_class_id
            && shape_addr == expected_shape_addr
            && expected_field_index < (*obj).field_count
            && plain_array_index_guard(expected_keys, expected_field_index, true)
            && object_key_matches_field(obj, key, expected_field_index)
            && (!require_raw_f64
                || (is_plain_number_bits(value_bits)
                    && crate::gc::layout_typed_raw_f64_slot_for_user(
                        object_addr,
                        expected_field_index as usize,
                    )))
            && !class_setter_in_chain(class_id, &key_name)
            && !descriptor_blocks_class_field_set(object_addr, class_id, &key_name);
        (shape_addr, class_id, gc_type, valid)
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_class_field_set_guard(
    site_id: u64,
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    key: *const crate::StringHeader,
    expected_field_index: u32,
    value: f64,
    require_raw_f64: i32,
) -> i32 {
    let value_bits = value.to_bits();
    let (shape_addr, class_id, gc_type, contract_valid) = class_field_set_contract(
        receiver,
        expected_class_id,
        expected_keys,
        key,
        expected_field_index,
        require_raw_f64 != 0,
        value_bits,
    );
    let object_addr = normalize_raw_object_addr(receiver.to_bits());
    let observation = Observation {
        source: ObservationSource::Property,
        object_addr: shape_keyed_object_addr(ObservationSource::Property, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: expected_field_index as u64,
        value_tag: stable_value_kind(value_bits),
    };
    if guard_observe(
        site_id,
        TypedFeedbackSiteKind::PropertySet,
        observation,
        contract_valid,
    ) {
        1
    } else {
        0
    }
}

fn shape_keyed_object_addr(source: ObservationSource, object_addr: usize) -> usize {
    if matches!(
        source,
        ObservationSource::Property | ObservationSource::Method | ObservationSource::NumericWrite
    ) {
        0
    } else {
        object_addr
    }
}

#[no_mangle]
pub unsafe extern "C" fn js_typed_feedback_native_call_method(
    site_id: u64,
    object: f64,
    method_name_ptr: *const i8,
    method_name_len: usize,
    args_ptr: *const f64,
    args_len: usize,
) -> f64 {
    let bits = object.to_bits();
    let object_addr = normalize_raw_object_addr(bits);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let name_hash = if valid_method_name(method_name_ptr, method_name_len) {
        hash_bytes(std::slice::from_raw_parts(
            method_name_ptr as *const u8,
            method_name_len,
        ))
    } else {
        0
    };
    let observation = Observation {
        source: ObservationSource::Method,
        object_addr: shape_keyed_object_addr(ObservationSource::Method, object_addr),
        shape_addr,
        key_hash: name_hash,
        class_id,
        heap_type: gc_type,
        aux: 0,
        value_tag: value_tag(bits),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::MethodCall,
        observation,
        valid_method_name(method_name_ptr, method_name_len)
            && bits != TAG_NULL
            && bits != TAG_UNDEFINED,
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::object::js_native_call_method(
        object,
        method_name_ptr,
        method_name_len,
        args_ptr,
        args_len,
    )
}

#[no_mangle]
pub unsafe extern "C" fn js_typed_feedback_native_call_method_apply(
    site_id: u64,
    object: f64,
    method_name_ptr: *const i8,
    method_name_len: usize,
    args_array: i64,
) -> f64 {
    let bits = object.to_bits();
    let object_addr = normalize_raw_object_addr(bits);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let name_hash = if valid_method_name(method_name_ptr, method_name_len) {
        hash_bytes(std::slice::from_raw_parts(
            method_name_ptr as *const u8,
            method_name_len,
        ))
    } else {
        0
    };
    let observation = Observation {
        source: ObservationSource::Method,
        object_addr: shape_keyed_object_addr(ObservationSource::Method, object_addr),
        shape_addr,
        key_hash: name_hash,
        class_id,
        heap_type: gc_type,
        aux: 0,
        value_tag: value_tag(bits),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::MethodCall,
        observation,
        valid_method_name(method_name_ptr, method_name_len)
            && bits != TAG_NULL
            && bits != TAG_UNDEFINED,
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::object::js_native_call_method_apply(object, method_name_ptr, method_name_len, args_array)
}

#[no_mangle]
pub unsafe extern "C" fn js_typed_feedback_method_direct_call_guard(
    site_id: u64,
    receiver: f64,
    expected_class_id: u32,
    expected_keys: *const ArrayHeader,
    method_name_ptr: *const i8,
    method_name_len: usize,
    expected_func_ptr: *const u8,
) -> i32 {
    let bits = receiver.to_bits();
    let (shape_addr, class_id, gc_type, name_hash, contract_valid) = method_direct_call_contract(
        receiver,
        expected_class_id,
        expected_keys,
        method_name_ptr,
        method_name_len,
        expected_func_ptr,
    );
    let object_addr = normalize_raw_object_addr(bits);
    let observation = Observation {
        source: ObservationSource::Method,
        object_addr: shape_keyed_object_addr(ObservationSource::Method, object_addr),
        shape_addr,
        key_hash: name_hash,
        class_id,
        heap_type: gc_type,
        aux: expected_func_ptr as u64,
        value_tag: value_tag(bits),
    };
    if guard_observe(
        site_id,
        TypedFeedbackSiteKind::MethodCall,
        observation,
        contract_valid,
    ) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_closure_direct_call_guard(
    site_id: u64,
    closure_value: f64,
    expected_func_ptr: *const u8,
    expected_arity: u32,
    call_arity: u32,
) -> i32 {
    let bits = closure_value.to_bits();
    let raw_ptr = if (bits & TAG_MASK) == POINTER_TAG {
        (bits & POINTER_MASK) as *const crate::closure::ClosureHeader
    } else if (bits >> 48) == 0 && bits >= 0x10000 {
        bits as *const crate::closure::ClosureHeader
    } else {
        std::ptr::null()
    };
    let closure_ptr = crate::closure::clean_closure_ptr(raw_ptr);
    let func_ptr = crate::closure::get_valid_func_ptr(closure_ptr);
    let has_rest = !func_ptr.is_null() && crate::closure::lookup_closure_rest(func_ptr).is_some();
    let declared = if func_ptr.is_null() {
        None
    } else {
        crate::closure::lookup_closure_arity(func_ptr)
    };
    let contract_valid = !expected_func_ptr.is_null()
        && !func_ptr.is_null()
        && func_ptr == expected_func_ptr
        && func_ptr != crate::closure::BOUND_METHOD_FUNC_PTR
        && !has_rest
        && declared.unwrap_or(expected_arity) == expected_arity
        && expected_arity == call_arity;
    let observation = Observation {
        source: ObservationSource::Closure,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id: 0,
        heap_type: if func_ptr.is_null() {
            0
        } else {
            crate::gc::GC_TYPE_CLOSURE as u16
        },
        aux: func_ptr as u64,
        value_tag: stable_value_kind(bits),
    };
    if guard_observe(
        site_id,
        TypedFeedbackSiteKind::ClosureCall,
        observation,
        contract_valid,
    ) {
        1
    } else {
        0
    }
}

fn observe_array(site_id: u64, arr: *const ArrayHeader, index: u32) {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let (class_id, heap_type, aux, element_kind) = classify_array(raw_addr, Some(index));
    observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        Observation {
            source: ObservationSource::Array,
            object_addr: 0,
            shape_addr: 0,
            key_hash: 0,
            class_id,
            heap_type,
            aux,
            value_tag: element_kind,
        },
    );
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_get_f64(
    site_id: u64,
    arr: *const ArrayHeader,
    index: u32,
) -> f64 {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let (class_id, heap_type, aux, element_kind) = classify_array(raw_addr, Some(index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: element_kind,
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        plain_array_index_guard(arr, index, true),
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::array::js_array_get_f64(arr, index)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_plain_array_index_get_guard(
    site_id: u64,
    receiver: f64,
    index_value: f64,
    index: i32,
    require_in_bounds: i32,
) -> i32 {
    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    let observed_index = if index >= 0 { index as u32 } else { u32::MAX };
    let (class_id, heap_type, aux, element_kind) = classify_array(raw_addr, Some(observed_index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: element_kind,
    };
    let contract_valid = is_plain_number_bits(index_value.to_bits())
        && index >= 0
        && plain_array_index_guard(
            raw_addr as *const ArrayHeader,
            index as u32,
            require_in_bounds != 0,
        );
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        contract_valid,
    );
    if pass {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_numeric_array_index_get_guard(
    site_id: u64,
    receiver: f64,
    index_value: f64,
    index: i32,
    require_in_bounds: i32,
) -> i32 {
    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    let observed_index = if index >= 0 { index as u32 } else { u32::MAX };
    let (class_id, heap_type, aux, element_kind) = classify_array(raw_addr, Some(observed_index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: element_kind,
    };
    let contract_valid = is_plain_number_bits(index_value.to_bits())
        && index >= 0
        && numeric_array_index_guard(
            raw_addr as *const ArrayHeader,
            index as u32,
            require_in_bounds != 0,
        );
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        contract_valid,
    );
    if pass {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_index_get_fallback_boxed(
    site_id: u64,
    receiver: f64,
    index: f64,
) -> f64 {
    record_fallback_call(site_id);

    let receiver_value = crate::value::JSValue::from_bits(receiver.to_bits());
    if receiver_value.is_string() || receiver_value.is_short_string() {
        return crate::value::js_dyn_index_get(receiver, index);
    }

    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    if raw_addr == 0 {
        return f64::from_bits(TAG_UNDEFINED);
    }

    if crate::buffer::is_registered_buffer(raw_addr)
        || crate::typedarray::lookup_typed_array_kind(raw_addr).is_some()
        || crate::set::is_registered_set(raw_addr)
        || crate::map::is_registered_map(raw_addr)
    {
        if !index.is_finite() || index < 0.0 {
            return f64::from_bits(TAG_UNDEFINED);
        }
        return crate::array::js_array_get_f64(raw_addr as *const ArrayHeader, index as u32);
    }

    if !crate::object::is_valid_obj_ptr(raw_addr as *const u8) {
        return f64::from_bits(TAG_UNDEFINED);
    }

    unsafe {
        let gc_header =
            (raw_addr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader;
        match (*gc_header).obj_type {
            crate::gc::GC_TYPE_ARRAY | crate::gc::GC_TYPE_LAZY_ARRAY => {
                if !index.is_finite() || index < 0.0 {
                    f64::from_bits(TAG_UNDEFINED)
                } else {
                    crate::array::js_array_get_f64(raw_addr as *const ArrayHeader, index as u32)
                }
            }
            crate::gc::GC_TYPE_OBJECT | crate::gc::GC_TYPE_CLOSURE => {
                let key_ptr = index_value_to_property_key(index);
                crate::object::js_object_get_field_by_name_f64(
                    raw_addr as *const ObjectHeader,
                    key_ptr,
                )
            }
            _ => f64::from_bits(TAG_UNDEFINED),
        }
    }
}

fn index_value_to_property_key(index: f64) -> *const crate::StringHeader {
    let bits = index.to_bits();
    let tag = bits & TAG_MASK;
    if tag == STRING_TAG || tag == SHORT_STRING_TAG {
        return crate::value::js_get_string_pointer_unified(index) as *const crate::StringHeader;
    }

    let key = if index.is_nan() {
        "NaN".to_string()
    } else if index.is_infinite() {
        if index.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        }
    } else {
        let int_index = index as i32;
        if index == int_index as f64 {
            int_index.to_string()
        } else {
            format!("{}", index)
        }
    };
    crate::string::js_string_from_bytes(key.as_ptr(), key.len() as u32)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_set_f64(
    site_id: u64,
    arr: *mut ArrayHeader,
    index: u32,
    value: f64,
) {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let (class_id, heap_type, aux, _element_kind) = classify_array(raw_addr, Some(index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        plain_array_index_guard(arr, index, true),
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::array::js_array_set_f64(arr, index, value);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_set_f64_extend(
    site_id: u64,
    arr: *mut ArrayHeader,
    index: u32,
    value: f64,
) -> *mut ArrayHeader {
    let raw_addr = normalize_raw_object_addr(arr as u64);
    let (class_id, heap_type, aux, _element_kind) = classify_array(raw_addr, Some(index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        plain_array_index_guard(arr, index, false),
    );
    if !pass {
        record_fallback_call(site_id);
    }
    crate::array::js_array_set_f64_extend(arr, index, value)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_plain_array_index_set_guard(
    site_id: u64,
    receiver: f64,
    index: i32,
    value: f64,
    require_in_bounds: i32,
) -> i32 {
    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    let observed_index = if index >= 0 { index as u32 } else { u32::MAX };
    let (class_id, heap_type, aux, _element_kind) = classify_array(raw_addr, Some(observed_index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let contract_valid = index >= 0
        && plain_array_index_guard(
            raw_addr as *const ArrayHeader,
            index as u32,
            require_in_bounds != 0,
        );
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        contract_valid,
    );
    if pass {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_numeric_array_index_set_guard(
    site_id: u64,
    receiver: f64,
    index: i32,
    value: f64,
    require_in_bounds: i32,
) -> i32 {
    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    let observed_index = if index >= 0 { index as u32 } else { u32::MAX };
    let (class_id, heap_type, aux, _element_kind) = classify_array(raw_addr, Some(observed_index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let contract_valid = index >= 0
        && is_numeric_value_bits(value.to_bits())
        && numeric_array_index_guard(
            raw_addr as *const ArrayHeader,
            index as u32,
            require_in_bounds != 0,
        );
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        contract_valid,
    );
    if pass {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_numeric_array_push_guard(
    site_id: u64,
    receiver: f64,
    value: f64,
) -> i32 {
    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    let push_index = match gc_header_for_user_addr(raw_addr) {
        Some(header) if unsafe { (*header).obj_type == crate::gc::GC_TYPE_ARRAY } => unsafe {
            (*(raw_addr as *const ArrayHeader)).length
        },
        _ => u32::MAX,
    };
    let (class_id, heap_type, aux, _element_kind) = classify_array(raw_addr, Some(push_index));
    let observation = Observation {
        source: ObservationSource::Array,
        object_addr: 0,
        shape_addr: 0,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::ArrayElement,
        observation,
        numeric_array_push_guard(raw_addr as *const ArrayHeader, value),
    );
    if pass {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_index_set_fallback_boxed(
    site_id: u64,
    receiver: f64,
    index: i32,
    value: f64,
) -> f64 {
    record_fallback_call(site_id);

    let raw_addr = normalize_raw_object_addr(receiver.to_bits());
    if raw_addr == 0 {
        return receiver;
    }

    let index_value = index as f64;
    if crate::buffer::is_registered_buffer(raw_addr)
        || crate::typedarray::lookup_typed_array_kind(raw_addr).is_some()
    {
        crate::array::js_array_set_index_or_string(
            raw_addr as *mut ArrayHeader,
            index_value,
            value,
        );
        return receiver;
    }

    if !crate::object::is_valid_obj_ptr(raw_addr as *const u8) {
        return receiver;
    }

    unsafe {
        let gc_header =
            (raw_addr as *const u8).sub(crate::gc::GC_HEADER_SIZE) as *const crate::gc::GcHeader;
        match (*gc_header).obj_type {
            crate::gc::GC_TYPE_ARRAY | crate::gc::GC_TYPE_LAZY_ARRAY => {
                let new_arr = crate::array::js_array_set_index_or_string(
                    raw_addr as *mut ArrayHeader,
                    index_value,
                    value,
                );
                crate::value::js_nanbox_pointer(new_arr as i64)
            }
            crate::gc::GC_TYPE_OBJECT | crate::gc::GC_TYPE_CLOSURE => {
                let key = index.to_string();
                let key_ptr = crate::string::js_string_from_bytes(key.as_ptr(), key.len() as u32);
                crate::object::js_object_set_field_by_name(
                    raw_addr as *mut ObjectHeader,
                    key_ptr,
                    value,
                );
                receiver
            }
            _ => receiver,
        }
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_observe_array_element(
    site_id: u64,
    arr: *const ArrayHeader,
    index: u32,
) {
    observe_array(site_id, arr, index);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_set_string_key(
    site_id: u64,
    arr: *mut ArrayHeader,
    key: *const crate::StringHeader,
    value: f64,
) -> *mut ArrayHeader {
    observe_array(site_id, arr, u32::MAX);
    record_guard_fail(site_id);
    record_fallback_call(site_id);
    crate::array::js_array_set_string_key(arr, key, value)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_array_set_index_or_string(
    site_id: u64,
    arr: *mut ArrayHeader,
    idx: f64,
    value: f64,
) -> *mut ArrayHeader {
    let index = if idx.is_finite() && idx >= 0.0 && idx <= u32::MAX as f64 {
        idx as u32
    } else {
        u32::MAX
    };
    observe_array(site_id, arr, index);
    if index == u32::MAX {
        record_guard_fail(site_id);
        record_fallback_call(site_id);
    } else {
        record_guard_pass(site_id);
    }
    crate::array::js_array_set_index_or_string(arr, idx, value)
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_object_set_index_polymorphic(
    site_id: u64,
    obj_handle: i64,
    idx: f64,
    value: f64,
) {
    let index = if idx.is_finite() && idx >= 0.0 && idx <= u32::MAX as f64 {
        idx as u32
    } else {
        u32::MAX
    };
    observe_array(site_id, obj_handle as *const ArrayHeader, index);
    record_guard_fail(site_id);
    record_fallback_call(site_id);
    crate::object::js_object_set_index_polymorphic(obj_handle, idx, value);
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_object_set_unboxed_f64_field(
    site_id: u64,
    obj: *mut ObjectHeader,
    field_index: u32,
    key: *const crate::StringHeader,
    value: f64,
) {
    let object_addr = normalize_raw_object_addr(obj as u64);
    let (shape_addr, class_id, gc_type) = object_shape(object_addr);
    let observation = Observation {
        source: ObservationSource::NumericWrite,
        object_addr: shape_keyed_object_addr(ObservationSource::NumericWrite, object_addr),
        shape_addr,
        key_hash: key_hash(key),
        class_id,
        heap_type: gc_type,
        aux: field_index as u64,
        value_tag: stable_value_kind(value.to_bits()),
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::NumericFieldWrite,
        observation,
        object_key_matches_field(obj, key, field_index) && is_plain_number_bits(value.to_bits()),
    );
    if pass {
        crate::object::js_object_set_unboxed_f64_field(obj, field_index, value);
    } else {
        record_fallback_call(site_id);
        crate::object::js_object_set_field_by_name(obj, key, value);
    }
}

#[no_mangle]
pub extern "C" fn js_typed_feedback_observe_helper_return(site_id: u64, value: f64) -> f64 {
    let bits = value.to_bits();
    let (shape_addr, class_id, heap_type, aux, value_kind) = helper_return_facts(bits);
    let observation = Observation {
        source: ObservationSource::HelperReturn,
        object_addr: 0,
        shape_addr,
        key_hash: 0,
        class_id,
        heap_type,
        aux,
        value_tag: value_kind,
    };
    let pass = guard_observe(
        site_id,
        TypedFeedbackSiteKind::HelperReturn,
        observation,
        true,
    );
    if !pass {
        record_fallback_call(site_id);
    }
    value
}

pub(crate) fn invalidate_shape_change(
    obj: *mut ObjectHeader,
    old_shape: *mut ArrayHeader,
    new_shape: *mut ArrayHeader,
) {
    if old_shape == new_shape {
        return;
    }
    let obj_addr = obj as usize;
    let (_, class_id, _) = object_shape(obj_addr);
    let old_addr = old_shape as usize;
    let new_addr = new_shape as usize;
    let mut reg = registry();
    reg.shape_invalidations = reg.shape_invalidations.saturating_add(1);
    for site in reg.sites.values_mut() {
        let affected = site
            .observations
            .iter()
            .any(|obs| obs.affected_by_shape_change(old_addr, new_addr, class_id));
        if affected {
            site.shape_invalidations = site.shape_invalidations.saturating_add(1);
        }
    }
}

pub(crate) fn invalidate_method_change(class_id: u32) {
    let mut reg = registry();
    reg.method_invalidations = reg.method_invalidations.saturating_add(1);
    for site in reg.sites.values_mut() {
        if site.metadata.kind == TypedFeedbackSiteKind::MethodCall
            && site
                .observations
                .iter()
                .any(|obs| class_id == 0 || obs.class_id == class_id)
        {
            site.method_invalidations = site.method_invalidations.saturating_add(1);
        }
    }
}

pub(crate) fn invalidate_representation_change(obj_addr: usize) {
    if obj_addr == 0 {
        return;
    }
    let (shape_addr, class_id, heap_type) = object_shape(obj_addr);
    let mut reg = registry();
    reg.representation_invalidations = reg.representation_invalidations.saturating_add(1);
    for site in reg.sites.values_mut() {
        if site.observations.iter().any(|obs| {
            obs.affected_by_representation_change(obj_addr, shape_addr, class_id, heap_type)
        }) {
            site.representation_invalidations = site.representation_invalidations.saturating_add(1);
        }
    }
}

pub fn scan_typed_feedback_roots_mut(visitor: &mut crate::gc::RuntimeRootVisitor<'_>) {
    let mut reg = registry();
    for site in reg.sites.values_mut() {
        for obs in &mut site.observations {
            if obs.roots_object_addr() {
                visitor.visit_usize_slot(&mut obs.object_addr);
            }
            if obs.roots_shape_addr() {
                visitor.visit_usize_slot(&mut obs.shape_addr);
            }
        }
    }
}

pub fn typed_feedback_snapshot() -> TypedFeedbackSnapshot {
    let reg = registry();
    let mut snapshot = TypedFeedbackSnapshot {
        total_sites: reg.sites.len(),
        shape_invalidations: reg.shape_invalidations,
        method_invalidations: reg.method_invalidations,
        representation_invalidations: reg.representation_invalidations,
        ..TypedFeedbackSnapshot::default()
    };
    let mut rows = Vec::with_capacity(reg.sites.len());
    for site in reg.sites.values() {
        let state = site.state();
        *snapshot
            .by_kind
            .entry(site.metadata.kind.as_str().to_string())
            .or_insert(0) += 1;
        *snapshot
            .by_state
            .entry(state.as_str().to_string())
            .or_insert(0) += 1;
        rows.push(TypedFeedbackSiteSnapshot {
            site_id: site.site_id,
            kind: site.metadata.kind.as_str(),
            state: state.as_str(),
            module: site.metadata.module.clone(),
            function: site.metadata.function.clone(),
            source_label: site.metadata.source_label.clone(),
            operation: site.metadata.operation.clone(),
            guard_name: site.metadata.guard_name.clone(),
            fallback_name: site.metadata.fallback_name.clone(),
            observed_count: site.observed_count,
            observation_count: site.observations.len(),
            guard_passes: site.guard_passes,
            guard_failures: site.guard_failures,
            fallback_calls: site.fallback_calls,
            shape_invalidations: site.shape_invalidations,
            method_invalidations: site.method_invalidations,
            representation_invalidations: site.representation_invalidations,
        });
        snapshot.guard_passes = snapshot.guard_passes.saturating_add(site.guard_passes);
        snapshot.guard_failures = snapshot.guard_failures.saturating_add(site.guard_failures);
        snapshot.fallback_calls = snapshot.fallback_calls.saturating_add(site.fallback_calls);
        snapshot
            .guards_by_name
            .entry(site.metadata.guard_name.clone())
            .or_insert(GuardCounterSnapshot {
                passes: 0,
                failures: 0,
                fallback_calls: 0,
            })
            .add_site(site);
    }
    rows.sort_by_key(|row| row.site_id);
    snapshot.sites = rows;
    snapshot
}

pub fn typed_feedback_trace_json() -> serde_json::Value {
    let snapshot = typed_feedback_snapshot();
    serde_json::json!({
        "total_sites": snapshot.total_sites,
        "by_kind": snapshot.by_kind,
        "by_state": snapshot.by_state,
        "invalidations": {
            "shape": snapshot.shape_invalidations,
            "method": snapshot.method_invalidations,
            "representation": snapshot.representation_invalidations,
        },
        "guards": {
            "passes": snapshot.guard_passes,
            "failures": snapshot.guard_failures,
            "fallback_calls": snapshot.fallback_calls,
            "by_guard": snapshot.guards_by_name.iter().map(|(name, counters)| {
                (
                    name.clone(),
                    serde_json::json!({
                        "passes": counters.passes,
                        "failures": counters.failures,
                        "fallback_calls": counters.fallback_calls,
                    }),
                )
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
        },
        "sites": snapshot.sites.iter().map(|site| {
            serde_json::json!({
                "site_id": site.site_id,
                "kind": site.kind,
                "state": site.state,
                "module": site.module,
                "function": site.function,
                "source_label": site.source_label,
                "operation": site.operation,
                "guard_name": site.guard_name,
                "fallback_name": site.fallback_name,
                "observed_count": site.observed_count,
                "observation_count": site.observation_count,
                "guard_passes": site.guard_passes,
                "guard_failures": site.guard_failures,
                "fallback_calls": site.fallback_calls,
                "guards": {
                    "passes": site.guard_passes,
                    "failures": site.guard_failures,
                    "fallback_calls": site.fallback_calls,
                },
                "invalidations": {
                    "shape": site.shape_invalidations,
                    "method": site.method_invalidations,
                    "representation": site.representation_invalidations,
                },
            })
        }).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
pub(crate) fn reset_typed_feedback_for_tests() {
    let mut reg = registry();
    *reg = TypedFeedbackRegistry::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    static CLASS_FIELD_SETTER_CALLS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    static CLASS_FIELD_SETTER_VALUE_BITS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    static CLASS_FIELD_GETTER_CALLS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    extern "C" fn test_class_field_setter(_this: f64, value: f64) -> f64 {
        CLASS_FIELD_SETTER_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        CLASS_FIELD_SETTER_VALUE_BITS.store(value.to_bits(), std::sync::atomic::Ordering::SeqCst);
        f64::from_bits(crate::value::TAG_UNDEFINED)
    }

    extern "C" fn test_class_field_getter(_this: f64) -> f64 {
        CLASS_FIELD_GETTER_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        99.0
    }

    extern "C" fn test_direct_method(_this: f64, value: f64) -> f64 {
        value
    }

    extern "C" fn test_direct_closure(
        _closure: *const crate::closure::ClosureHeader,
        arg: f64,
    ) -> f64 {
        arg
    }

    fn test_direct_closure_ptr() -> *const u8 {
        test_direct_closure as *const () as *const u8
    }

    fn test_direct_method_ptr() -> *const u8 {
        test_direct_method as *const () as *const u8
    }

    fn register(site_id: u64, kind: TypedFeedbackSiteKind, op: &'static str) {
        js_typed_feedback_register_site(
            site_id,
            kind as u32,
            b"typed_feedback_test.ts".as_ptr(),
            "typed_feedback_test.ts".len(),
            b"probe".as_ptr(),
            "probe".len(),
            op.as_ptr(),
            op.len(),
            op.as_ptr(),
            op.len(),
            b"test_guard".as_ptr(),
            "test_guard".len(),
            b"test_fallback".as_ptr(),
            "test_fallback".len(),
        );
    }

    fn class_instance(
        class_id: u32,
        key_name: &'static [u8],
    ) -> (
        *mut ObjectHeader,
        *mut ArrayHeader,
        *const crate::StringHeader,
        f64,
    ) {
        let mut packed = Vec::with_capacity(key_name.len() + 1);
        packed.extend_from_slice(key_name);
        packed.push(0);
        let obj = crate::object::js_object_alloc_class_with_keys(
            class_id,
            0,
            1,
            packed.as_ptr(),
            packed.len() as u32,
        );
        let key = crate::string::js_string_from_bytes(key_name.as_ptr(), key_name.len() as u32);
        let keys = unsafe { (*obj).keys_array };
        let receiver = crate::value::js_nanbox_pointer(obj as i64);
        (obj, keys, key, receiver)
    }

    unsafe fn register_test_method(class_id: u32, name: &'static [u8]) {
        crate::object::js_register_class_method(
            class_id as i64,
            name.as_ptr(),
            name.len() as i64,
            test_direct_method as *const () as usize as i64,
            1,
        );
    }

    fn plain_object_with_key(
        key_name: &'static [u8],
    ) -> (*mut ObjectHeader, *const crate::StringHeader) {
        let obj = crate::object::js_object_alloc(0, 0);
        let key = crate::string::js_string_from_bytes(key_name.as_ptr(), key_name.len() as u32);
        (obj, key)
    }

    #[test]
    fn typed_feedback_registers_source_attribution() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(1, TypedFeedbackSiteKind::PropertyGet, "obj.x");
        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.total_sites, 1);
        assert_eq!(snapshot.by_kind["property_get"], 1);
        assert_eq!(snapshot.by_state["uninitialized"], 1);
        assert_eq!(snapshot.sites[0].module, "typed_feedback_test.ts");
        assert_eq!(snapshot.sites[0].function, "probe");
        assert_eq!(snapshot.sites[0].operation, "obj.x");
    }

    #[test]
    fn typed_feedback_state_transitions_to_megamorphic() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(2, TypedFeedbackSiteKind::HelperReturn, "helper");
        for i in 0..POLYMORPHIC_CAP {
            observe(
                2,
                TypedFeedbackSiteKind::HelperReturn,
                Observation {
                    source: ObservationSource::HelperReturn,
                    object_addr: 0,
                    shape_addr: 0,
                    key_hash: 0,
                    class_id: 0,
                    heap_type: 0,
                    aux: i as u64,
                    value_tag: i as u16,
                },
            );
        }
        assert_eq!(typed_feedback_snapshot().sites[0].state, "polymorphic");
        observe(
            2,
            TypedFeedbackSiteKind::HelperReturn,
            Observation {
                source: ObservationSource::HelperReturn,
                object_addr: 0,
                shape_addr: 0,
                key_hash: 0,
                class_id: 0,
                heap_type: 0,
                aux: 99,
                value_tag: 99,
            },
        );
        assert_eq!(typed_feedback_snapshot().sites[0].state, "megamorphic");
    }

    #[test]
    fn typed_feedback_invalidation_counters_are_site_attributed() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(3, TypedFeedbackSiteKind::MethodCall, "m");
        observe(
            3,
            TypedFeedbackSiteKind::MethodCall,
            Observation {
                source: ObservationSource::Method,
                object_addr: 0,
                shape_addr: 0,
                key_hash: 1,
                class_id: 42,
                heap_type: 0,
                aux: 1,
                value_tag: 0,
            },
        );
        invalidate_method_change(42);
        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.method_invalidations, 1);
        assert_eq!(snapshot.sites[0].method_invalidations, 1);
    }

    #[test]
    fn typed_feedback_property_and_method_keys_ignore_receiver_identity() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(5, TypedFeedbackSiteKind::PropertyGet, "obj.x");
        register(6, TypedFeedbackSiteKind::MethodCall, "obj.m()");
        for object_addr in [0x1000_0000usize, 0x2000_0000usize] {
            observe(
                5,
                TypedFeedbackSiteKind::PropertyGet,
                Observation {
                    source: ObservationSource::Property,
                    object_addr,
                    shape_addr: 0xCAFE,
                    key_hash: 0xA11C_E,
                    class_id: 7,
                    heap_type: crate::gc::GC_TYPE_OBJECT as u16,
                    aux: 0,
                    value_tag: 0,
                },
            );
            observe(
                6,
                TypedFeedbackSiteKind::MethodCall,
                Observation {
                    source: ObservationSource::Method,
                    object_addr,
                    shape_addr: 0xCAFE,
                    key_hash: 0xBEE,
                    class_id: 7,
                    heap_type: crate::gc::GC_TYPE_OBJECT as u16,
                    aux: 0,
                    value_tag: value_tag(POINTER_TAG),
                },
            );
        }

        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.by_state["monomorphic"], 2);
        assert!(snapshot
            .sites
            .iter()
            .all(|site| site.observed_count == 2 && site.observation_count == 1));
    }

    #[test]
    fn typed_feedback_array_keys_use_element_facts_not_sample_identity() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(7, TypedFeedbackSiteKind::ArrayElement, "arr[i]");

        let values1 = [1.0, 1.5];
        let values2 = [2.0, 2.5, 3.0, 3.5];
        let arr1 = crate::array::js_array_from_f64(values1.as_ptr(), values1.len() as u32);
        let arr2 = crate::array::js_array_from_f64(values2.as_ptr(), values2.len() as u32);

        js_typed_feedback_observe_array_element(7, arr1, 0);
        js_typed_feedback_observe_array_element(7, arr2, 3);

        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.sites[0].state, "monomorphic");
        assert_eq!(snapshot.sites[0].observed_count, 2);
        assert_eq!(snapshot.sites[0].observation_count, 1);

        let reg = registry();
        let observation = reg.sites.get(&7).unwrap().observations[0];
        assert_eq!(observation.object_addr, 0);
        assert_eq!(observation.heap_type, crate::gc::GC_TYPE_ARRAY as u16);
        assert_eq!(observation.value_tag, STABLE_VALUE_NUMBER);
    }

    #[test]
    fn typed_feedback_helper_return_keys_use_shape_facts_not_sample_identity() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(8, TypedFeedbackSiteKind::HelperReturn, "helper()");

        let packed = b"x\0";
        let obj1 = crate::object::js_object_alloc_with_shape(
            0x7EED_0008,
            1,
            packed.as_ptr(),
            packed.len() as u32,
        );
        let obj2 = crate::object::js_object_alloc_with_shape(
            0x7EED_0008,
            1,
            packed.as_ptr(),
            packed.len() as u32,
        );

        js_typed_feedback_observe_helper_return(8, crate::value::js_nanbox_pointer(obj1 as i64));
        js_typed_feedback_observe_helper_return(8, crate::value::js_nanbox_pointer(obj2 as i64));

        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.sites[0].state, "monomorphic");
        assert_eq!(snapshot.sites[0].observed_count, 2);
        assert_eq!(snapshot.sites[0].observation_count, 1);

        let reg = registry();
        let observation = reg.sites.get(&8).unwrap().observations[0];
        assert_eq!(observation.object_addr, 0);
        assert_eq!(observation.heap_type, crate::gc::GC_TYPE_OBJECT as u16);
        assert_ne!(observation.shape_addr, 0);
    }

    #[test]
    fn typed_feedback_tracks_all_site_categories() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        let kinds = [
            TypedFeedbackSiteKind::PropertyGet,
            TypedFeedbackSiteKind::PropertySet,
            TypedFeedbackSiteKind::MethodCall,
            TypedFeedbackSiteKind::ClosureCall,
            TypedFeedbackSiteKind::ArrayElement,
            TypedFeedbackSiteKind::NumericFieldWrite,
            TypedFeedbackSiteKind::HelperReturn,
        ];
        for (idx, kind) in kinds.iter().copied().enumerate() {
            register(10 + idx as u64, kind, kind.as_str());
        }

        let snapshot = typed_feedback_snapshot();
        assert_eq!(snapshot.total_sites, kinds.len());
        for kind in kinds {
            assert_eq!(snapshot.by_kind[kind.as_str()], 1);
        }
    }

    #[test]
    fn typed_feedback_unboxed_numeric_write_falls_back_for_string_values() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(21, TypedFeedbackSiteKind::NumericFieldWrite, "obj.x=");

        let packed = b"x\0";
        let obj = crate::object::js_object_alloc_with_shape(
            0x7EED_0021,
            1,
            packed.as_ptr(),
            packed.len() as u32,
        );
        let key = crate::string::js_string_from_bytes(b"x".as_ptr(), 1);

        js_typed_feedback_object_set_unboxed_f64_field(21, obj, 0, key, 1.0);
        let payload = crate::string::js_string_from_bytes(b"fallback".as_ptr(), 8);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        js_typed_feedback_object_set_unboxed_f64_field(21, obj, 0, key, payload_value);

        let stored = crate::object::js_object_get_field_by_name_f64(obj, key);
        assert_eq!(stored.to_bits(), payload_value.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_helper_return_guard_failure_returns_original_value() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(22, TypedFeedbackSiteKind::HelperReturn, "helper()");

        let first = js_typed_feedback_observe_helper_return(22, 42.0);
        assert_eq!(first.to_bits(), 42.0f64.to_bits());

        let payload = crate::string::js_string_from_bytes(b"shape-change".as_ptr(), 12);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        let second = js_typed_feedback_observe_helper_return(22, payload_value);
        assert_eq!(second.to_bits(), payload_value.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_array_guard_failure_matches_jsvalue_fallback() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(23, TypedFeedbackSiteKind::ArrayElement, "arr[i]");

        let values = [1.0, 2.0];
        let arr = crate::array::js_array_from_f64(values.as_ptr(), values.len() as u32);
        let expected = crate::array::js_array_get_f64(arr, 5);
        let actual = js_typed_feedback_array_get_f64(23, arr, 5);
        assert_eq!(actual.to_bits(), expected.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_array_get_guard_failure_uses_jsvalue_object_fallback() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(25, TypedFeedbackSiteKind::ArrayElement, "arr[i]");

        let obj = crate::object::js_object_alloc(0, 0);
        let obj_box = f64::from_bits(crate::value::JSValue::pointer(obj as *const u8).bits());
        let key = crate::string::js_string_from_bytes(b"0".as_ptr(), 1);
        crate::object::js_object_set_field_by_name(obj, key, 42.0);

        // Models an array-typed compiled read whose receiver was replaced by
        // a dynamic object at a JS boundary. The guard must reject it before
        // codegen reads ArrayHeader fields; fallback then performs obj["0"].
        let guard = js_typed_feedback_plain_array_index_get_guard(25, obj_box, 0.0, 0, 1);
        assert_eq!(guard, 0);

        let actual = js_typed_feedback_array_index_get_fallback_boxed(25, obj_box, 0.0);
        assert_eq!(actual.to_bits(), 42.0f64.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_numeric_array_get_guard_requires_numeric_layout() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(26, TypedFeedbackSiteKind::ArrayElement, "arr[i]");

        let values = [1.0, 2.0];
        let arr = crate::array::js_array_from_f64(values.as_ptr(), values.len() as u32);
        let arr_box = crate::value::js_nanbox_pointer(arr as i64);

        let first = js_typed_feedback_numeric_array_index_get_guard(26, arr_box, 0.0, 0, 1);
        assert_eq!(first, 1);

        let payload = crate::string::js_string_from_bytes(b"downgraded".as_ptr(), 10);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        crate::array::js_array_set_f64(arr, 0, payload_value);
        assert_eq!(crate::array::js_array_is_numeric_f64_layout(arr), 0);

        let second = js_typed_feedback_numeric_array_index_get_guard(26, arr_box, 0.0, 0, 1);
        assert_eq!(second, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 0);
    }

    #[test]
    fn typed_feedback_numeric_array_set_guard_requires_numeric_value_and_layout() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(27, TypedFeedbackSiteKind::ArrayElement, "arr[i]=");

        let values = [1.0, 2.0];
        let arr = crate::array::js_array_from_f64(values.as_ptr(), values.len() as u32);
        let arr_box = crate::value::js_nanbox_pointer(arr as i64);

        let first = js_typed_feedback_numeric_array_index_set_guard(27, arr_box, 1, 3.0, 1);
        assert_eq!(first, 1);

        let payload = crate::string::js_string_from_bytes(b"not-number".as_ptr(), 10);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        let nonnumeric =
            js_typed_feedback_numeric_array_index_set_guard(27, arr_box, 1, payload_value, 1);
        assert_eq!(nonnumeric, 0);

        crate::array::js_array_set_f64(arr, 0, payload_value);
        let downgraded = js_typed_feedback_numeric_array_index_set_guard(27, arr_box, 1, 4.0, 1);
        assert_eq!(downgraded, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 2);
        assert_eq!(site.fallback_calls, 0);
    }

    #[test]
    fn typed_feedback_numeric_array_push_guard_requires_room_numeric_value_and_layout() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(28, TypedFeedbackSiteKind::ArrayElement, "arr.push");

        let arr = crate::array::js_array_alloc(0);
        let arr_box = crate::value::js_nanbox_pointer(arr as i64);

        let first = js_typed_feedback_numeric_array_push_guard(28, arr_box, 1.0);
        assert_eq!(first, 1);

        let payload = crate::string::js_string_from_bytes(b"not-number".as_ptr(), 10);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        let nonnumeric = js_typed_feedback_numeric_array_push_guard(28, arr_box, payload_value);
        assert_eq!(nonnumeric, 0);

        let capacity = unsafe { (*arr).capacity };
        for i in 0..capacity {
            crate::array::js_array_push_f64(arr, i as f64);
        }
        let full = js_typed_feedback_numeric_array_push_guard(28, arr_box, 2.0);
        assert_eq!(full, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 2);
        assert_eq!(site.fallback_calls, 0);
    }

    #[test]
    fn typed_feedback_non_bounded_array_set_guard_failure_uses_jsvalue_object_fallback() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(24, TypedFeedbackSiteKind::ArrayElement, "arr[i]=");

        let obj = crate::object::js_object_alloc(0, 0);
        let obj_box = f64::from_bits(crate::value::JSValue::pointer(obj as *const u8).bits());

        // Models an array-typed compiled local slot that receives an object
        // from a dynamic boundary: the non-bounded set guard must fail before
        // codegen can read ArrayHeader fields or raw-store an element.
        let guard = js_typed_feedback_plain_array_index_set_guard(24, obj_box, 0, 99.0, 0);
        assert_eq!(guard, 0);

        let returned = js_typed_feedback_array_index_set_fallback_boxed(24, obj_box, 0, 99.0);
        assert_eq!(returned.to_bits(), obj_box.to_bits());

        let key = crate::string::js_string_from_bytes(b"0".as_ptr(), 1);
        let stored = crate::object::js_object_get_field_by_name_f64(obj, key);
        assert_eq!(stored.to_bits(), 99.0f64.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_class_field_set_guard_fails_for_frozen_object() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(31, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0031;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));
        crate::object::js_object_freeze(receiver);

        let guard =
            js_typed_feedback_class_field_set_guard(31, receiver, class_id, keys, key, 0, 2.0, 0);
        assert_eq!(guard, 0);
        assert_eq!(
            crate::object::js_object_get_field(obj, 0).bits(),
            1.0f64.to_bits()
        );

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 0);
    }

    #[test]
    fn typed_feedback_class_field_set_guard_allows_sealed_writable_field() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(37, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0037;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));
        crate::object::js_object_seal(receiver);

        let guard =
            js_typed_feedback_class_field_set_guard(37, receiver, class_id, keys, key, 0, 2.0, 0);
        assert_eq!(guard, 1);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 0);
    }

    #[test]
    fn typed_feedback_class_field_set_guard_falls_back_for_class_setter() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        CLASS_FIELD_SETTER_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
        CLASS_FIELD_SETTER_VALUE_BITS.store(0, std::sync::atomic::Ordering::SeqCst);
        register(32, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0032;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));
        unsafe {
            crate::object::js_register_class_setter(
                class_id as i64,
                b"x".as_ptr(),
                1,
                test_class_field_setter as *const () as usize as i64,
            );
        }

        let guard =
            js_typed_feedback_class_field_set_guard(32, receiver, class_id, keys, key, 0, 7.0, 0);
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(32);
        crate::object::js_object_set_field_by_name(obj, key, 7.0);

        assert_eq!(
            CLASS_FIELD_SETTER_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            CLASS_FIELD_SETTER_VALUE_BITS.load(std::sync::atomic::Ordering::SeqCst),
            7.0f64.to_bits()
        );
        assert_eq!(
            crate::object::js_object_get_field(obj, 0).bits(),
            1.0f64.to_bits()
        );

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_class_field_get_guard_falls_back_for_class_getter() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        CLASS_FIELD_GETTER_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
        register(33, TypedFeedbackSiteKind::PropertyGet, "obj.x");

        let class_id = 0x7EED_0033;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));
        unsafe {
            crate::object::js_register_class_getter(
                class_id as i64,
                b"x".as_ptr(),
                1,
                test_class_field_getter as *const () as usize as i64,
            );
        }

        let guard =
            js_typed_feedback_class_field_get_guard(33, receiver, class_id, keys, key, 0, 0);
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(33);
        let value = crate::object::js_object_get_field_by_name_f64(obj, key);
        assert_eq!(value.to_bits(), 1.0f64.to_bits());
        assert_eq!(
            CLASS_FIELD_GETTER_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_class_field_set_guard_falls_back_for_non_writable_descriptor() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(34, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0034;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));
        crate::object::set_property_attrs(
            obj as usize,
            "x".to_string(),
            crate::object::PropertyAttrs::new(false, true, true),
        );

        let guard =
            js_typed_feedback_class_field_set_guard(34, receiver, class_id, keys, key, 0, 7.0, 0);
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(34);
        assert_eq!(
            crate::object::js_object_get_field(obj, 0).bits(),
            1.0f64.to_bits()
        );

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_class_field_guards_fall_back_for_prototype_accessor_descriptor() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(35, TypedFeedbackSiteKind::PropertyGet, "obj.x");
        register(36, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0035;
        let (obj, keys, key, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(1.0f64.to_bits()));

        let proto = crate::object::js_object_alloc(0, 0);
        crate::object::set_accessor_descriptor(
            proto as usize,
            "x".to_string(),
            crate::object::AccessorDescriptor { get: 0, set: 0 },
        );
        {
            let mut prototypes = crate::object::CLASS_PROTOTYPE_OBJECTS.write().unwrap();
            if prototypes.is_none() {
                *prototypes = Some(std::collections::HashMap::new());
            }
            prototypes
                .as_mut()
                .unwrap()
                .insert(class_id, proto as usize);
        }

        let get_guard =
            js_typed_feedback_class_field_get_guard(35, receiver, class_id, keys, key, 0, 0);
        let set_guard =
            js_typed_feedback_class_field_set_guard(36, receiver, class_id, keys, key, 0, 7.0, 0);
        assert_eq!(get_guard, 0);
        assert_eq!(set_guard, 0);

        let snapshot = typed_feedback_snapshot();
        let get_site = snapshot
            .sites
            .iter()
            .find(|site| site.site_id == 35)
            .unwrap();
        let set_site = snapshot
            .sites
            .iter()
            .find(|site| site.site_id == 36)
            .unwrap();
        assert_eq!(get_site.guard_failures, 1);
        assert_eq!(set_site.guard_failures, 1);
    }

    #[test]
    fn typed_feedback_class_field_get_guard_falls_back_after_shape_transition() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(39, TypedFeedbackSiteKind::PropertyGet, "obj.x");

        let class_id = 0x7EED_0039;
        let (obj, expected_keys, key_x, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_field(obj, 0, crate::JSValue::from_bits(5.0f64.to_bits()));
        let first = js_typed_feedback_class_field_get_guard(
            39,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            0,
        );
        assert_eq!(first, 1);

        let key_y = crate::string::js_string_from_bytes(b"y".as_ptr(), 1);
        crate::object::js_object_set_field_by_name(obj, key_y, 10.0);
        assert_ne!(unsafe { (*obj).keys_array }, expected_keys);

        let second = js_typed_feedback_class_field_get_guard(
            39,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            0,
        );
        assert_eq!(second, 0);
        js_typed_feedback_record_fallback_call(39);
        let stored = crate::object::js_object_get_field_by_name_f64(obj, key_x);
        assert_eq!(stored.to_bits(), 5.0f64.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_class_field_get_guard_requires_raw_f64_layout_when_requested() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(43, TypedFeedbackSiteKind::PropertyGet, "obj.x");

        let class_id = 0x7EED_0043;
        let (obj, expected_keys, key_x, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_unboxed_f64_field(obj, 0, 5.0);
        let raw_mask = [0b1u64];
        crate::gc::js_gc_init_typed_shape_layout(
            obj as u64,
            1,
            raw_mask.as_ptr(),
            raw_mask.len() as u32,
            std::ptr::null(),
            0,
        );

        let first = js_typed_feedback_class_field_get_guard(
            43,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            1,
        );
        assert_eq!(first, 1);

        let payload = crate::string::js_string_from_bytes(b"boxed".as_ptr(), 5);
        crate::object::js_object_set_field(obj, 0, crate::JSValue::string_ptr(payload));

        let second = js_typed_feedback_class_field_get_guard(
            43,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            1,
        );
        assert_eq!(second, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
        assert!(site.representation_invalidations >= 1);
    }

    #[test]
    fn typed_feedback_class_field_set_guard_requires_raw_f64_value_and_layout() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(44, TypedFeedbackSiteKind::PropertySet, "obj.x=");

        let class_id = 0x7EED_0044;
        let (obj, expected_keys, key_x, receiver) = class_instance(class_id, b"x");
        crate::object::js_object_set_unboxed_f64_field(obj, 0, 1.0);
        let raw_mask = [0b1u64];
        crate::gc::js_gc_init_typed_shape_layout(
            obj as u64,
            1,
            raw_mask.as_ptr(),
            raw_mask.len() as u32,
            std::ptr::null(),
            0,
        );

        let first = js_typed_feedback_class_field_set_guard(
            44,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            2.0,
            1,
        );
        assert_eq!(first, 1);

        let payload = crate::string::js_string_from_bytes(b"boxed".as_ptr(), 5);
        let payload_value = crate::value::js_nanbox_string(payload as i64);
        let second = js_typed_feedback_class_field_set_guard(
            44,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            payload_value,
            1,
        );
        assert_eq!(second, 0);

        let short = crate::value::JSValue::try_short_string(b"abc").unwrap();
        let third = js_typed_feedback_class_field_set_guard(
            44,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            f64::from_bits(short.bits()),
            1,
        );
        assert_eq!(third, 0);

        let handle_value = f64::from_bits(crate::value::JS_HANDLE_TAG | 0x1234);
        let fourth = js_typed_feedback_class_field_set_guard(
            44,
            receiver,
            class_id,
            expected_keys,
            key_x,
            0,
            handle_value,
            1,
        );
        assert_eq!(fourth, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 3);
    }

    #[test]
    fn typed_feedback_object_set_fast_hits_learned_dynamic_key_transition() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(34, TypedFeedbackSiteKind::PropertySet, "obj[dyn]=");

        let (first_obj, key) = plain_object_with_key(b"dyn_fast_key_34");
        js_typed_feedback_object_set_field_by_name_fast(34, first_obj, key, 11.0);
        let first_site = &typed_feedback_snapshot().sites[0];
        assert_eq!(first_site.fallback_calls, 1);

        let second_obj = crate::object::js_object_alloc(0, 0);
        js_typed_feedback_object_set_field_by_name_fast(34, second_obj, key, 12.0);
        let stored = crate::object::js_object_get_field_by_name_f64(second_obj, key);
        assert_eq!(stored.to_bits(), 12.0f64.to_bits());

        let site = &typed_feedback_snapshot().sites[0];
        if crate::object::descriptors_in_use() {
            assert_eq!(site.fallback_calls, 2);
        } else {
            assert_eq!(site.fallback_calls, 1);
            assert!(site.guard_passes >= 1);
        }
    }

    #[test]
    fn typed_feedback_object_set_fast_falls_back_for_uncached_dynamic_key() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(35, TypedFeedbackSiteKind::PropertySet, "obj[dyn_miss]=");

        let (obj, key) = plain_object_with_key(b"dyn_uncached_key_35");
        js_typed_feedback_object_set_field_by_name_fast(35, obj, key, 21.0);

        let stored = crate::object::js_object_get_field_by_name_f64(obj, key);
        assert_eq!(stored.to_bits(), 21.0f64.to_bits());
        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_method_direct_guard_passes_for_exact_registered_method() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(61, TypedFeedbackSiteKind::MethodCall, "obj.m()");

        let class_id = 0x7EED_0061;
        let (_, keys, _, receiver) = class_instance(class_id, b"x");
        unsafe { register_test_method(class_id, b"m") };

        let guard = unsafe {
            js_typed_feedback_method_direct_call_guard(
                61,
                receiver,
                class_id,
                keys,
                b"m".as_ptr() as *const i8,
                1,
                test_direct_method_ptr(),
            )
        };
        assert_eq!(guard, 1);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 0);
        assert_eq!(site.fallback_calls, 0);
        assert_eq!(site.state, "monomorphic");
    }

    #[test]
    fn typed_feedback_method_direct_guard_fails_for_own_method_replacement() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(62, TypedFeedbackSiteKind::MethodCall, "obj.m()");

        let class_id = 0x7EED_0062;
        let (obj, keys, _, receiver) = class_instance(class_id, b"x");
        unsafe { register_test_method(class_id, b"m") };
        let key_m = crate::string::js_string_from_bytes(b"m".as_ptr(), 1);
        crate::object::js_object_set_field_by_name(obj, key_m, 123.0);

        let guard = unsafe {
            js_typed_feedback_method_direct_call_guard(
                62,
                receiver,
                class_id,
                keys,
                b"m".as_ptr() as *const i8,
                1,
                test_direct_method_ptr(),
            )
        };
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(62);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_method_direct_guard_fails_for_prototype_method_registration() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(63, TypedFeedbackSiteKind::MethodCall, "obj.m()");

        let class_id = 0x7EED_0063;
        let (_, keys, _, receiver) = class_instance(class_id, b"x");
        unsafe {
            register_test_method(class_id, b"m");
            crate::object::js_register_prototype_method(
                class_id,
                b"m".as_ptr(),
                1,
                f64::from_bits(crate::value::TAG_UNDEFINED),
            );
        }

        let guard = unsafe {
            js_typed_feedback_method_direct_call_guard(
                63,
                receiver,
                class_id,
                keys,
                b"m".as_ptr() as *const i8,
                1,
                test_direct_method_ptr(),
            )
        };
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(63);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 0);
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_method_direct_guard_fails_for_native_receiver() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(64, TypedFeedbackSiteKind::MethodCall, "native.m()");

        let native = crate::object::js_object_alloc(crate::object::NATIVE_MODULE_CLASS_ID, 0);
        let receiver = crate::value::js_nanbox_pointer(native as i64);

        let guard = unsafe {
            js_typed_feedback_method_direct_call_guard(
                64,
                receiver,
                crate::object::NATIVE_MODULE_CLASS_ID,
                std::ptr::null(),
                b"m".as_ptr() as *const i8,
                1,
                test_direct_method_ptr(),
            )
        };
        assert_eq!(guard, 0);
        js_typed_feedback_record_fallback_call(64);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_failures, 1);
        assert_eq!(site.fallback_calls, 1);
    }

    #[test]
    fn typed_feedback_method_direct_guard_fails_after_megamorphic_site() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(65, TypedFeedbackSiteKind::MethodCall, "obj.m()");
        for i in 0..=POLYMORPHIC_CAP {
            observe(
                65,
                TypedFeedbackSiteKind::MethodCall,
                Observation {
                    source: ObservationSource::Method,
                    object_addr: 0,
                    shape_addr: 0x1000 + i,
                    key_hash: i as u64,
                    class_id: i as u32 + 1,
                    heap_type: crate::gc::GC_TYPE_OBJECT as u16,
                    aux: i as u64,
                    value_tag: STABLE_VALUE_POINTER,
                },
            );
        }

        let class_id = 0x7EED_0065;
        let (_, keys, _, receiver) = class_instance(class_id, b"x");
        unsafe { register_test_method(class_id, b"m") };
        let guard = unsafe {
            js_typed_feedback_method_direct_call_guard(
                65,
                receiver,
                class_id,
                keys,
                b"m".as_ptr() as *const i8,
                1,
                test_direct_method_ptr(),
            )
        };
        assert_eq!(guard, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.state, "megamorphic");
        assert_eq!(site.guard_failures, 1);
    }

    #[test]
    fn typed_feedback_closure_direct_guard_passes_and_rejects_bound_sentinel() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(66, TypedFeedbackSiteKind::ClosureCall, "cb()");

        let fn_ptr = test_direct_closure_ptr();
        crate::closure::js_register_closure_arity(fn_ptr, 1);
        let closure = crate::closure::js_closure_alloc_singleton(fn_ptr);
        let closure_value = crate::value::js_nanbox_pointer(closure as i64);
        let pass = js_typed_feedback_closure_direct_call_guard(66, closure_value, fn_ptr, 1, 1);
        assert_eq!(pass, 1);

        let bound = crate::closure::js_closure_alloc(crate::closure::BOUND_METHOD_FUNC_PTR, 0);
        let bound_value = crate::value::js_nanbox_pointer(bound as i64);
        let fail = js_typed_feedback_closure_direct_call_guard(66, bound_value, fn_ptr, 1, 1);
        assert_eq!(fail, 0);

        let site = &typed_feedback_snapshot().sites[0];
        assert_eq!(site.guard_passes, 1);
        assert_eq!(site.guard_failures, 1);
    }

    #[test]
    fn typed_feedback_trace_json_reports_counts() {
        let _guard = TYPED_FEEDBACK_TEST_LOCK.lock().unwrap();
        reset_typed_feedback_for_tests();
        register(4, TypedFeedbackSiteKind::ArrayElement, "arr[i]");
        js_typed_feedback_record_guard_pass(4);
        js_typed_feedback_record_guard_fail(4);
        js_typed_feedback_record_fallback_call(4);
        let json = typed_feedback_trace_json();
        assert_eq!(json["total_sites"].as_u64(), Some(1));
        assert_eq!(json["by_kind"]["array_element"].as_u64(), Some(1));
        assert_eq!(json["by_state"]["uninitialized"].as_u64(), Some(1));
        assert_eq!(json["guards"]["passes"].as_u64(), Some(1));
        assert_eq!(json["guards"]["failures"].as_u64(), Some(1));
        assert_eq!(json["guards"]["fallback_calls"].as_u64(), Some(1));
        assert_eq!(
            json["guards"]["by_guard"]["test_guard"]["fallback_calls"].as_u64(),
            Some(1)
        );
        assert_eq!(json["sites"][0]["guard_name"].as_str(), Some("test_guard"));
        assert_eq!(
            json["sites"][0]["fallback_name"].as_str(),
            Some("test_fallback")
        );
        assert_eq!(json["sites"][0]["guard_passes"].as_u64(), Some(1));
        assert_eq!(json["sites"][0]["guard_failures"].as_u64(), Some(1));
        assert_eq!(json["sites"][0]["fallback_calls"].as_u64(), Some(1));
    }
}
