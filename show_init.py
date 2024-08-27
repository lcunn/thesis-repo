import json
import os

# Get the path to the current VSCode workspace settings
vscode_settings_path = os.path.join(os.getcwd(), ".vscode", "settings.json")

def toggle_init_visibility():
    if not os.path.exists(vscode_settings_path):
        print("VSCode settings file not found.")
        return

    # Load the current settings
    with open(vscode_settings_path, "r") as f:
        settings = json.load(f)

    # Toggle the visibility of __init__.py
    files_exclude = settings.get("files.exclude", {})
    init_pattern = "**/__init__.py"

    if init_pattern in files_exclude:
        # If the pattern is there, remove it (i.e., show the files)
        del files_exclude[init_pattern]
        print("Showing all __init__.py files.")
    else:
        # Otherwise, hide them
        files_exclude[init_pattern] = True
        print("Hiding all __init__.py files.")

    # Save the updated settings
    settings["files.exclude"] = files_exclude
    with open(vscode_settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    print("VSCode settings updated.")

if __name__ == "__main__":
    toggle_init_visibility()

