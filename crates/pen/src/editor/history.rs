//! Deterministic Undo / Redo command stack for the interactive `pen edit` scene editor.

use pen_scene::{Component, NodeId, Scene, SceneEnvironment, SceneNode, Transform};

/// Reversible scene edit command.
#[derive(Clone, Debug)]
pub enum EditCommand {
    SetTransform { node: NodeId, before: Transform, after: Transform },
    SetVisible { node: NodeId, before: bool, after: bool },
    AddNode { node: SceneNode },
    RemoveNode { index: usize, node: SceneNode },
    SetLightIntensity { node: NodeId, before: f32, after: f32 },
    SetEnvironment { before: SceneEnvironment, after: SceneEnvironment },
}

impl EditCommand {
    pub fn apply(&self, scene: &mut Scene) {
        match self {
            EditCommand::SetTransform { node, after, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    n.transform = *after;
                }
            }
            EditCommand::SetVisible { node, after, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    n.visible = *after;
                }
            }
            EditCommand::AddNode { node } => {
                scene.nodes.push(node.clone());
            }
            EditCommand::RemoveNode { node, .. } => {
                scene.nodes.retain(|n| n.id != node.id);
            }
            EditCommand::SetLightIntensity { node, after, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    for c in &mut n.components {
                        if let Component::Light { intensity, .. } = c {
                            *intensity = *after;
                        }
                    }
                }
            }
            EditCommand::SetEnvironment { after, .. } => {
                scene.environment = after.clone();
            }
        }
    }

    pub fn revert(&self, scene: &mut Scene) {
        match self {
            EditCommand::SetTransform { node, before, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    n.transform = *before;
                }
            }
            EditCommand::SetVisible { node, before, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    n.visible = *before;
                }
            }
            EditCommand::AddNode { node } => {
                scene.nodes.retain(|n| n.id != node.id);
            }
            EditCommand::RemoveNode { index, node } => {
                let pos = (*index).min(scene.nodes.len());
                scene.nodes.insert(pos, node.clone());
            }
            EditCommand::SetLightIntensity { node, before, .. } => {
                if let Some(n) = scene.find_node_mut(*node) {
                    for c in &mut n.components {
                        if let Component::Light { intensity, .. } = c {
                            *intensity = *before;
                        }
                    }
                }
            }
            EditCommand::SetEnvironment { before, .. } => {
                scene.environment = before.clone();
            }
        }
    }
}

/// Bounded undo/redo command history stack.
#[derive(Default)]
pub struct UndoHistory {
    undo_stack: Vec<EditCommand>,
    redo_stack: Vec<EditCommand>,
}

impl UndoHistory {
    pub fn execute(&mut self, scene: &mut Scene, cmd: EditCommand) {
        cmd.apply(scene);
        self.undo_stack.push(cmd);
        if self.undo_stack.len() > 256 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn push_already_applied(&mut self, cmd: EditCommand) {
        self.undo_stack.push(cmd);
        if self.undo_stack.len() > 256 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, scene: &mut Scene) -> bool {
        let Some(cmd) = self.undo_stack.pop() else { return false };
        cmd.revert(scene);
        self.redo_stack.push(cmd);
        true
    }

    pub fn redo(&mut self, scene: &mut Scene) -> bool {
        let Some(cmd) = self.redo_stack.pop() else { return false };
        cmd.apply(scene);
        self.undo_stack.push(cmd);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}
