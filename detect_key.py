#!/usr/bin/env python3
"""
Quick key detector - Press any key to see what it is
"""

from pynput import keyboard

def on_press(key):
    print(f"Key pressed: {key}")
    try:
        print(f"  - Has attribute 'char': {hasattr(key, 'char')}")
        if hasattr(key, 'char'):
            print(f"  - char: {key.char}")
        print(f"  - Has attribute 'name': {hasattr(key, 'name')}")
        if hasattr(key, 'name'):
            print(f"  - name: {key.name}")
        print(f"  - Type: {type(key)}")
        print(f"  - Repr: {repr(key)}")
    except Exception as e:
        print(f"  - Error: {e}")
    print()

def on_release(key):
    if key == keyboard.Key.esc:
        print("ESC pressed - exiting!")
        return False

print("=" * 60)
print("Key Detector")
print("=" * 60)
print("\nPress the key you want to use for recording.")
print("Press ESC to quit.\n")
print("=" * 60)
print()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
