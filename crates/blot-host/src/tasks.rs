//! Cooperative, synchronous Blot task ABI 2. No Wasm stack is suspended here.
//! Concurrent host callbacks are rejected at admission; resources use newer
//! guest ABI versions which this host does not admit. Identity never wraps.
use super::State;
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use wasmtime::{AsContextMut, Caller, Func, FuncType, Instance, Linker, Module, Store, Val, ValType};

pub const MODULE: &str = "blot:task/1";
const LIMIT: usize = 4096;
pub const IMPORTS: &[(&str, usize, bool)] = &[
    ("open", 0, true),
    ("close", 1, false),
    ("spawn", 3, true),
    ("join", 2, true),
    ("collect", 3, false),
    ("cancel", 1, false),
    ("poll", 2, true),
    ("drop", 2, false),
    ("drain", 0, false),
    ("checkpoint", 0, false),
    ("current", 0, true),
    ("resume", 1, false),
    ("transfer", 1, false),
];
fn exports(count: u32) -> Vec<(String, usize, bool)> {
    let mut result = vec![];
    for role in ["create", "save", "restore", "drop"] {
        result.push((format!("blot:task_context/{role}"), 1, true));
    }
    result.push(("blot:task_retain".into(), 1, false));
    result.push(("blot:task_release".into(), 1, false));
    for id in 0..count {
        for (role, result_value) in [("start", true), ("move_captures", false), ("move_result", false)] {
            result.push((format!("blot:task_{role}/{id}"), 1, result_value));
        }
    }
    result
}
fn signature(ty: &FuncType, parameters: usize, result: bool) -> bool {
    ty.params().len() == parameters
        && ty.params().all(|t| t.is_i32())
        && ty.results().len() == usize::from(result)
        && ty.results().all(|t| t.is_i32())
}
pub fn validate(module: &Module, count: u32) -> Result<()> {
    let mut seen = BTreeSet::new();
    for import in module.imports().filter(|i| i.module() == MODULE) {
        let (_, parameters, result) = IMPORTS
            .iter()
            .find(|(name, _, _)| *name == import.name())
            .ok_or_else(|| anyhow!("invalid_abi: unknown task import {}", import.name()))?;
        ensure!(seen.insert(import.name().to_owned()), "invalid_abi: repeated task import");
        ensure!(
            import.ty().func().is_some_and(|ty| signature(ty, *parameters, *result)),
            "invalid_abi: task import signature"
        );
    }
    for (name, _, _) in IMPORTS {
        ensure!(*name == "collect" || seen.contains(*name), "invalid_abi: missing task import {name}");
    }
    for (name, parameters, result) in exports(count) {
        ensure!(
            module
                .get_export(&name)
                .and_then(|e| e.func().cloned())
                .is_some_and(|ty| signature(&ty, parameters, result)),
            "invalid_abi: task export signature {name}"
        );
    }
    Ok(())
}
#[derive(Clone)]
struct Task {
    group: u32,
    parent: u32,
    native: i32,
    descriptor: u32,
    captures: i32,
    status: u32,
    consumed: bool,
    cancel: bool,
    result: Option<i32>,
    failure: Option<String>,
}
struct Group {
    owner: u32,
    children: Vec<u32>,
    closing: bool,
}
pub(super) struct Tasks {
    hooks: BTreeMap<String, Func>,
    count: u32,
    next: u32,
    active: u32,
    root: Option<i32>,
    tasks: BTreeMap<u32, Task>,
    groups: BTreeMap<u32, Group>,
    cleaning: bool,
}
impl Tasks {
    fn identity(&mut self) -> Result<u32> {
        ensure!(self.tasks.len() + self.groups.len() < LIMIT, "task_budget: too many live task records");
        let id = self.next;
        self.next = self.next.checked_add(1).ok_or_else(|| anyhow!("task_budget: generation exhausted"))?;
        Ok(id)
    }
    fn task(&self, id: u32) -> Result<&Task> {
        self.tasks.get(&id).ok_or_else(|| anyhow!("task_host_contract: stale task {id}"))
    }
    fn group(&self, id: u32) -> Result<&Group> {
        self.groups.get(&id).ok_or_else(|| anyhow!("task_host_contract: stale group {id}"))
    }
    fn live(&self, id: u32, descriptor: u32) -> Result<()> {
        let task = self.task(id)?;
        ensure!(
            !task.consumed && task.descriptor == descriptor,
            "task_handle_consumed: invalid task handle or descriptor"
        );
        let mut ancestor = self.active;
        while ancestor != 0 {
            ensure!(ancestor != id, "task_host_contract: cannot join an ancestor");
            ancestor = self.task(ancestor)?.parent;
        }
        Ok(())
    }
}
fn state(context: &impl wasmtime::AsContext<Data = State>) -> &Tasks {
    context.as_context().data().tasks.as_ref().unwrap()
}
fn call(mut context: impl AsContextMut<Data = State>, name: &str, value: i32) -> Result<i32> {
    let function = state(&context).hooks.get(name).copied().ok_or_else(|| anyhow!("missing task hook {name}"))?;
    let has_result = function.ty(&context).results().len() != 0;
    let mut out = [Val::I32(0)];
    function.call(context.as_context_mut(), &[Val::I32(value)], if has_result { &mut out } else { &mut [] })?;
    Ok(out[0].i32().unwrap())
}
fn ownership(mut context: impl AsContextMut<Data = State>, enter: bool) -> Result<()> {
    let hooks = context.as_context().data().ownership.clone().unwrap();
    if enter {
        hooks.enter.call(context.as_context_mut(), ())?;
    } else {
        hooks.leave.call(context.as_context_mut(), 0)?;
    }
    Ok(())
}
fn activate(mut context: impl AsContextMut<Data = State>, id: u32) -> Result<()> {
    let runtime = state(&context);
    let native = if id == 0 { runtime.root.context("task invocation not started")? } else { runtime.task(id)?.native };
    call(context.as_context_mut(), "blot:task_context/restore", native)?;
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().active = id;
    Ok(())
}
pub(super) fn install(store: &mut Store<State>, instance: Instance, count: u32) -> Result<()> {
    let mut hooks = BTreeMap::new();
    for (name, _, _) in exports(count) {
        let function = instance.get_func(&mut *store, &name).context("missing task hook")?;
        hooks.insert(name, function);
    }
    store.data_mut().tasks = Some(Tasks {
        hooks,
        count,
        next: 1,
        active: 0,
        root: None,
        tasks: BTreeMap::new(),
        groups: BTreeMap::new(),
        cleaning: false,
    });
    Ok(())
}
pub(super) fn begin(mut context: impl AsContextMut<Data = State>) -> Result<()> {
    if context.as_context().data().tasks.is_none() {
        return Ok(());
    }
    if state(&context).root.is_none() {
        let native = call(context.as_context_mut(), "blot:task_context/create", 0)?;
        context.as_context_mut().data_mut().tasks.as_mut().unwrap().root = Some(native);
        call(context.as_context_mut(), "blot:task_context/save", native)?;
    }
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().active = 0;
    Ok(())
}
fn request(mut context: impl AsContextMut<Data = State>, id: u32) -> Result<()> {
    if state(&context).task(id)?.status >= 3 {
        return Ok(());
    }
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().cancel = true;
    let children: Vec<_> =
        state(&context).groups.values().filter(|g| g.owner == id).flat_map(|g| g.children.clone()).collect();
    for child in children {
        request(context.as_context_mut(), child)?;
    }
    Ok(())
}
fn failed(mut context: impl AsContextMut<Data = State>, id: u32, error: String) -> Result<()> {
    let group = state(&context).task(id)?.group;
    context
        .as_context_mut()
        .data_mut()
        .tasks
        .as_mut()
        .unwrap()
        .tasks
        .get_mut(&id)
        .unwrap()
        .failure
        .get_or_insert(error);
    let siblings = state(&context).group(group)?.children.clone();
    for sibling in siblings {
        if sibling != id {
            request(context.as_context_mut(), sibling)?;
        }
    }
    Ok(())
}
fn discard(mut context: impl AsContextMut<Data = State>, id: u32) -> Result<()> {
    let result = context.as_context_mut().data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().result.take();
    if let Some(result) = result {
        call(context.as_context_mut(), "blot:task_release", result)?;
    }
    Ok(())
}
fn start(mut context: impl AsContextMut<Data = State>, id: u32) -> Result<()> {
    let entry = state(&context).task(id)?.clone();
    if entry.status != 0 {
        return Ok(());
    }
    let prior = state(&context).active;
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().status = 1;
    activate(context.as_context_mut(), id)?;
    ownership(context.as_context_mut(), true)?;
    if !entry.cancel {
        let outcome = call(context.as_context_mut(), &format!("blot:task_start/{}", entry.descriptor), entry.captures);
        match outcome {
            Ok(result) => {
                match call(context.as_context_mut(), &format!("blot:task_move_result/{}", entry.descriptor), result) {
                    Ok(_) => {
                        context.as_context_mut().data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().result =
                            Some(result)
                    }
                    Err(error) => {
                        let _ = call(context.as_context_mut(), "blot:task_release", result);
                        failed(context.as_context_mut(), id, format!("{error:#}"))?;
                    }
                }
            }
            Err(error) => failed(context.as_context_mut(), id, format!("{error:#}"))?,
        }
    }
    let groups: Vec<_> = state(&context).groups.iter().filter(|(_, g)| g.owner == id).map(|(&key, _)| key).collect();
    for group in groups {
        if let Err(error) = close(context.as_context_mut(), group) {
            failed(context.as_context_mut(), id, format!("{error:#}"))?;
        }
    }
    // Cleanup is attempted fully, even after a trap. Native ownership context
    // stays live until captures, child scopes and the ownership frame retire.
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().cleaning = true;
    let release = call(context.as_context_mut(), "blot:task_release", entry.captures);
    let leave = ownership(context.as_context_mut(), false);
    let dropped = call(context.as_context_mut(), "blot:task_context/drop", entry.native);
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().cleaning = false;
    for error in [release.err(), leave.err(), dropped.err()].into_iter().flatten() {
        failed(context.as_context_mut(), id, format!("{error:#}"))?;
    }
    activate(context.as_context_mut(), prior)?;
    let task = state(&context).task(id)?;
    if (task.cancel || task.failure.is_some())
        && let Err(error) = discard(context.as_context_mut(), id)
    {
        failed(context.as_context_mut(), id, format!("{error:#}"))?;
    }
    let mut scope = context.as_context_mut();
    let runtime = scope.data_mut().tasks.as_mut().unwrap();
    let task = runtime.tasks.get_mut(&id).unwrap();
    task.status = if task.failure.is_some() {
        4
    } else if task.cancel {
        5
    } else {
        3
    };
    Ok(())
}
fn run(mut context: impl AsContextMut<Data = State>, group: u32) -> Result<()> {
    let mut index = 0;
    while index < state(&context).group(group)?.children.len() {
        let id = state(&context).group(group)?.children[index];
        let mut ancestor = state(&context).active;
        let mut skip = false;
        while ancestor != 0 {
            skip |= ancestor == id;
            ancestor = state(&context).task(ancestor)?.parent;
        }
        if !skip {
            start(context.as_context_mut(), id)?;
        }
        index += 1;
    }
    Ok(())
}
fn group_failure(context: &impl wasmtime::AsContext<Data = State>, group: u32) -> Option<String> {
    state(context).group(group).ok()?.children.iter().find_map(|id| state(context).task(*id).ok()?.failure.clone())
}
fn close(mut context: impl AsContextMut<Data = State>, group: u32) -> Result<()> {
    state(&context).group(group)?;
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().groups.get_mut(&group).unwrap().closing = true;
    run(context.as_context_mut(), group)?;
    let children = state(&context).group(group)?.children.clone();
    let mut failure = group_failure(&context, group);
    for id in children {
        if let Err(error) = discard(context.as_context_mut(), id) {
            failure.get_or_insert(format!("{error:#}"));
        }
        context.as_context_mut().data_mut().tasks.as_mut().unwrap().tasks.remove(&id);
    }
    context.as_context_mut().data_mut().tasks.as_mut().unwrap().groups.remove(&group);
    if let Some(error) = failure {
        bail!("task_failure: {error}");
    }
    Ok(())
}
pub(super) fn finish(mut context: impl AsContextMut<Data = State>, cancel: bool) -> Result<()> {
    if context.as_context().data().tasks.is_none() {
        return Ok(());
    }
    if state(&context).root.is_none() {
        return Ok(());
    }
    activate(context.as_context_mut(), 0)?;
    let groups: Vec<_> = state(&context).groups.iter().filter(|(_, g)| g.owner == 0).map(|(&id, _)| id).collect();
    if cancel {
        let children: Vec<_> = groups.iter().flat_map(|id| state(&context).groups[id].children.clone()).collect();
        for id in children {
            request(context.as_context_mut(), id)?;
        }
    }
    let mut failure = None;
    for group in groups {
        if let Err(error) = close(context.as_context_mut(), group) {
            failure.get_or_insert(error);
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}
pub(super) fn dispose(mut context: impl AsContextMut<Data = State>) -> Result<()> {
    if context.as_context().data().tasks.is_none() {
        return Ok(());
    }
    let finished = finish(context.as_context_mut(), true);
    let dropped = if let Some(root) = state(&context).root {
        call(context.as_context_mut(), "blot:task_context/drop", root).map(|_| ())
    } else {
        Ok(())
    };
    context.as_context_mut().data_mut().tasks = None;
    finished.and(dropped)
}
fn invoke(mut caller: Caller<'_, State>, name: &str, arguments: &[Val], results: &mut [Val]) -> Result<()> {
    let n = |i: usize| arguments[i].i32().unwrap().cast_unsigned();
    let active = state(&caller).active;
    let out = match name {
        "open" => {
            let runtime = caller.data_mut().tasks.as_mut().unwrap();
            let id = runtime.identity()?;
            runtime.groups.insert(id, Group { owner: active, children: vec![], closing: false });
            id
        }
        "spawn" => {
            let (group, descriptor, captures) = (n(0), n(1), n(2).cast_signed());
            let runtime = state(&caller);
            ensure!(descriptor < runtime.count, "task_host_contract: unknown descriptor");
            let target = runtime.group(group)?;
            if target.closing {
                let mut ancestor = active;
                let mut descendant = false;
                while ancestor != 0 {
                    let task = runtime.task(ancestor)?;
                    descendant |= task.group == group;
                    ancestor = task.parent;
                }
                ensure!(descendant, "task_escape: group is closing");
            }
            if active != 0 {
                ensure!(!runtime.task(active)?.cancel, "task_cancelled");
            }
            let id = caller.data_mut().tasks.as_mut().unwrap().identity()?;
            let native = call(&mut caller, "blot:task_context/create", 1)?;
            call(&mut caller, "blot:task_retain", captures)?;
            if let Err(error) = call(&mut caller, &format!("blot:task_move_captures/{descriptor}"), captures) {
                activate(&mut caller, active)?;
                let _ = call(&mut caller, "blot:task_release", captures);
                let _ = call(&mut caller, "blot:task_context/drop", native);
                return Err(error);
            }
            let runtime = caller.data_mut().tasks.as_mut().unwrap();
            runtime.tasks.insert(
                id,
                Task {
                    group,
                    parent: active,
                    native,
                    descriptor,
                    captures,
                    status: 0,
                    consumed: false,
                    cancel: false,
                    result: None,
                    failure: None,
                },
            );
            runtime.groups.get_mut(&group).unwrap().children.push(id);
            id
        }
        "join" => {
            let (id, descriptor) = (n(0), n(1));
            state(&caller).live(id, descriptor)?;
            let group = state(&caller).task(id)?.group;
            caller.data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().consumed = true;
            run(&mut caller, group)?;
            if let Some(error) = group_failure(&caller, group) {
                bail!("task_failure: {error}");
            }
            let result = caller
                .data_mut()
                .tasks
                .as_mut()
                .unwrap()
                .tasks
                .get_mut(&id)
                .unwrap()
                .result
                .take()
                .context("task_cancelled: no result")?;
            if let Err(error) = call(&mut caller, &format!("blot:task_move_result/{descriptor}"), result) {
                let _ = call(&mut caller, "blot:task_release", result);
                return Err(error);
            }
            result.cast_unsigned()
        }
        "collect" => {
            let (input, count, output) = (n(0) as usize, n(1) as usize, n(2) as usize);
            ensure!(count > 0 && count <= LIMIT, "task_host_contract: collection size");
            let memory = caller.data().arena.as_ref().context("task collection memory")?.memory;
            let bytes = memory.data(&caller);
            for (pointer, size) in [(input, count * 8), (output, count * 4)] {
                ensure!(
                    pointer != 0
                        && pointer.is_multiple_of(4)
                        && pointer.checked_add(size).is_some_and(|end| end <= bytes.len()),
                    "task_host_contract: collection range"
                );
            }
            let entries: Vec<_> = bytes[input..input + count * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|pair| {
                    (
                        u32::from_le_bytes(pair[..4].try_into().unwrap()),
                        u32::from_le_bytes(pair[4..].try_into().unwrap()),
                    )
                })
                .collect();
            let mut seen = BTreeSet::new();
            let mut groups = BTreeSet::new();
            for &(id, descriptor) in &entries {
                state(&caller).live(id, descriptor)?;
                ensure!(seen.insert(id), "task_handle_consumed: duplicate collection handle");
                groups.insert(state(&caller).task(id)?.group);
            }
            for &(id, _) in &entries {
                caller.data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().consumed = true;
            }
            for &group in &groups {
                run(&mut caller, group)?;
            }
            for &group in &groups {
                if let Some(error) = group_failure(&caller, group) {
                    bail!("task_failure: {error}");
                }
            }
            // Keep every result owned by its record until all adapters succeed.
            let mut values = vec![];
            for &(id, descriptor) in &entries {
                let value = state(&caller).task(id)?.result.context("task_cancelled: collected result absent")?;
                call(&mut caller, &format!("blot:task_move_result/{descriptor}"), value)?;
                values.push(value);
            }
            for (index, &(id, _)) in entries.iter().enumerate() {
                caller.data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().result = None;
                memory.write(&mut caller, output + index * 4, &values[index].to_le_bytes())?;
            }
            0
        }
        "cancel" | "drop" => {
            let id = n(0);
            let descriptor = if name == "drop" { n(1) } else { state(&caller).task(id)?.descriptor };
            state(&caller).live(id, descriptor)?;
            caller.data_mut().tasks.as_mut().unwrap().tasks.get_mut(&id).unwrap().consumed = true;
            request(&mut caller, id)?;
            start(&mut caller, id)?;
            discard(&mut caller, id)?;
            if let Some(error) = &state(&caller).task(id)?.failure {
                bail!("task_failure: {error}");
            }
            0
        }
        "poll" => {
            state(&caller).live(n(0), n(1))?;
            state(&caller).task(n(0))?.status
        }
        "close" => {
            close(&mut caller, n(0))?;
            0
        }
        "drain" => {
            let groups: Vec<_> =
                state(&caller).groups.iter().filter(|(_, g)| g.owner == active).map(|(&id, _)| id).collect();
            for group in groups {
                run(&mut caller, group)?;
                if let Some(error) = group_failure(&caller, group) {
                    bail!("task_failure: {error}");
                }
            }
            0
        }
        "current" => active,
        "resume" => {
            let id = n(0);
            ensure!(id == 0 || state(&caller).task(id)?.status == 1, "task_host_contract: stale native context");
            activate(&mut caller, id)?;
            0
        }
        "checkpoint" => {
            if active != 0 && !state(&caller).cleaning {
                ensure!(!state(&caller).task(active)?.cancel, "task_cancelled");
            }
            0
        }
        // This host admits only ABI 2/3, which cannot import resource journals.
        "transfer" => bail!("task_host_contract: resource token without a resource ABI"),
        _ => bail!("unknown task operation"),
    };
    if let Some(result) = results.first_mut() {
        *result = Val::I32(out.cast_signed());
    }
    Ok(())
}
pub(super) fn link(linker: &mut Linker<State>) -> Result<()> {
    for &(name, parameters, result) in IMPORTS {
        let ty = FuncType::new(
            linker.engine(),
            vec![ValType::I32; parameters],
            if result { vec![ValType::I32] } else { vec![] },
        );
        linker.func_new(MODULE, name, ty, move |caller, args, results| {
            invoke(caller, name, args, results).map_err(wasmtime::Error::from_anyhow)
        })?;
    }
    Ok(())
}
