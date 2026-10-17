"""An AT-SPI client working the game as a screen reader would (run.sh).

It finds the game on the accessibility bus, writes the tree as AT-SPI shows
it (tree-*.txt), and works it only through AT-SPI: it presses a toggle
(Action.DoAction) both ways, waits for the pause menu the run's script
opens, moves the focus (Component.GrabFocus) into Settings and presses it,
sets a slider (Value.CurrentValue), and once the script has closed the menu
moves the focus to the station's button. Each focus change it hears is written
as a screen reader would announce it (name, role, states, description) to
announcements.txt, which must match expected.txt line for line.
"""

import argparse
import os
import signal
import sys
import time

import gi

gi.require_version("Atspi", "2.0")
from gi.repository import Atspi, GLib  # noqa: E402

TITLE = "gdev"
announced = []


def say(line):
    print(f"atspi: {line}", flush=True)


def pump(seconds):
    """Lets AT-SPI's events in for `seconds`."""
    context = GLib.MainContext.default()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        while context.iteration(False):
            pass
        time.sleep(0.01)


def states(node):
    shown = []
    state_set = node.get_state_set()
    for state, word in [
        (Atspi.StateType.FOCUSED, "focused"),
        (Atspi.StateType.CHECKED, "checked"),
        (Atspi.StateType.PRESSED, "pressed"),
        (Atspi.StateType.EXPANDED, "expanded"),
        (Atspi.StateType.SELECTED, "selected"),
        (Atspi.StateType.MODAL, "modal"),
    ]:
        if state_set.contains(state):
            shown.append(word)
    return shown


def value_of(node):
    value = node.get_value_iface()
    return None if value is None else value.get_current_value()


def value_text(node):
    """What a screen reader says of a node's value: its text (AT-SPI 2.46's
    Value.Text, "MSAA 4x" for a choice), else its number."""
    value = node.get_value_iface()
    if value is None:
        return None
    text = value.get_text() if hasattr(value, "get_text") else None
    return text if text else f"{value.get_current_value():g}"


def announcement(node):
    """What a screen reader says as the focus lands on `node`."""
    parts = [node.get_name() or "(unnamed)", node.get_role_name()]
    parts += [state for state in states(node) if state != "focused"]
    value = value_text(node)
    if value is not None:
        parts.append(value)
    description = node.get_description()
    if description:
        parts.append(description)
    return ", ".join(parts)


def on_event(event):
    if event.type.startswith("object:state-changed:focused") and event.detail1 == 1:
        line = announcement(event.source)
        if not announced or announced[-1] != line:
            announced.append(line)
            say(f"announce: {line}")


def find_app():
    """The application whose window is the game's: AccessKit names an
    application after its executable (pen, or a package's gdev) and its
    window after the game's title."""
    desktop = Atspi.get_desktop(0)
    for index in range(desktop.get_child_count()):
        app = desktop.get_child_at_index(index)
        if app is None:
            continue
        for child in range(app.get_child_count()):
            window = app.get_child_at_index(child)
            if window is not None and window.get_name() == TITLE:
                return app
    return None


def walk(node, depth=0):
    yield node, depth
    for index in range(node.get_child_count()):
        child = node.get_child_at_index(index)
        if child is not None:
            yield from walk(child, depth + 1)


def describe(app):
    lines = []
    for node, depth in walk(app):
        line = f"{'  ' * depth}{node.get_role_name()} {node.get_name()!r}"
        shown = states(node)
        if shown:
            line += " [" + ",".join(shown) + "]"
        value = value_of(node)
        if value is not None:
            line += f" value={value:g}"
        action = node.get_action_iface()
        if action is not None and action.get_n_actions() > 0:
            line += " actions=" + ",".join(action.get_action_name(i) for i in range(action.get_n_actions()))
        lines.append(line)
    return "\n".join(lines) + "\n"


def find(app, name, role=None):
    app.clear_cache()
    for node, _ in walk(app):
        if node.get_name() == name and (role is None or node.get_role_name() == role):
            return node
    return None


def wait_for(what, check, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        found = check()
        if found:
            return found
        pump(0.05)
    raise SystemExit(f"atspi: FAILED: {what} within {seconds} s")


def press(node, name):
    action = node.get_action_iface()
    names = [action.get_action_name(i) for i in range(action.get_n_actions())]
    index = names.index(name) if name in names else 0
    action.do_action(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected")
    arguments = parser.parse_args()
    listener = Atspi.EventListener.new(on_event)
    listener.register("object:state-changed:focused")

    app = wait_for("the game on the accessibility bus", find_app, 120.0)
    say(f"found {TITLE!r} in application {app.get_name()!r} ({app.get_toolkit_name()})")
    # The editor's toggle, pressed through its action, takes the game
    # between editing and playing, both ways; it ends playing.
    toggle = wait_for("the editor's toggle", lambda: find(app, "Edit the level"), 30.0)
    on = lambda: bool({"checked", "pressed"} & set(states(find(app, "Edit the level"))))
    if not on():
        press(toggle, "click")
        wait_for("the editor's toggle to be on", on)
        say("editing")
    press(find(app, "Edit the level"), "click")
    wait_for("the editor's toggle to be off", lambda: not on())
    say("playing")
    # Playing: the HUD, the build bar, the station's button.
    wait_for("the station's button", lambda: find(app, "Hire five gnorps"), 10.0)
    with open(os.path.join(arguments.out, "tree-playing.txt"), "w") as file:
        file.write(describe(app))
    for name in ["Build", "Blueprints", "Stores", "Salt", "Gnorps", "Level editor"]:
        wait_for(f"{name!r} in the tree", lambda name=name: find(app, name))

    # The run's script opens the pause menu: a modal dialog, its first row focused.
    paused = wait_for("the pause menu", lambda: find(app, "Paused", "dialog"), 30.0)
    wait_for("Continue to have the focus", lambda: "focused" in states(find(app, "Continue")))
    with open(os.path.join(arguments.out, "tree-paused.txt"), "w") as file:
        file.write(describe(app))
    assert "modal" in states(paused), states(paused)
    find(app, "Settings").get_component_iface().grab_focus()
    wait_for("Settings to have the focus", lambda: "focused" in states(find(app, "Settings")))
    press(find(app, "Settings"), "click")
    wait_for("the settings", lambda: find(app, "Settings", "dialog"))
    fog = wait_for("the fog slider", lambda: find(app, "Fog"))
    fog.get_component_iface().grab_focus()
    wait_for("the slider to have the focus", lambda: "focused" in states(find(app, "Fog")))
    before = value_of(fog)
    fog.get_value_iface().set_current_value(1.5)
    wait_for("the fog to be 150%", lambda: abs((value_of(find(app, "Fog")) or 0) - 1.5) < 1e-3)
    say(f"fog set from {before:g} to 1.5")
    pump(0.3)
    with open(os.path.join(arguments.out, "tree-settings.txt"), "w") as file:
        file.write(describe(app))

    # The script backs out of the menu; playing again, the focus, moved as a
    # screen reader user moves it, reaches the game, which moves its own and
    # tells the bus.
    wait_for("the menu to close", lambda: find(app, "Hire five gnorps"), 30.0)
    find(app, "Hire five gnorps").get_component_iface().grab_focus()
    wait_for("the button to take the focus", lambda: "focused" in states(find(app, "Hire five gnorps")))
    pump(0.3)
    with open(os.path.join(arguments.out, "announcements.txt"), "w") as file:
        file.write("\n".join(announced) + "\n")
    os.kill(arguments.pid, signal.SIGTERM)

    if arguments.expected and os.path.exists(arguments.expected):
        with open(arguments.expected) as file:
            expected = [line for line in file.read().splitlines() if line and not line.startswith("#")]
        if announced != expected:
            say("FAILED: the announcements differ from expected.txt:")
            for line in announced:
                say(f"  heard:    {line}")
            for line in expected:
                say(f"  expected: {line}")
            return 1
    if arguments.expected and os.path.exists(arguments.expected):
        say(f"ok: {len(announced)} announcements, as expected.txt has them")
    else:
        say(f"ok: {len(announced)} announcements (no expected.txt to compare them with)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
