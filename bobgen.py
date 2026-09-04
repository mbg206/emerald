import tomllib
from os import mkdir, path, listdir, remove

from emerald.src.vers import VERSIONS

DATA_FOLDER = "bobdata"


if path.isdir(DATA_FOLDER):
    for file in listdir(DATA_FOLDER):
        remove(f"{DATA_FOLDER}/{file}")
else:
    mkdir(DATA_FOLDER)


with open("bob.toml", 'w', encoding="utf-8") as bob:
    first = True
    for file in listdir("emerald/"):
        if not file.endswith(".bot.toml"):
            continue
        filepath = f"emerald/{file}"

        id = file[0:-9]
        specfile = f"{id}.spec"
        runfile = f"{id}.py"

        if first:
            first = False
        else:
            bob.write("\n\n")

        with open(filepath, 'rb') as bot:
            botdata = tomllib.load(bot)

        bob.write(
            '[[config]]\n' +
            f'project_name = "{botdata['settings']['name']}"\n' +
            f'bot_configs = ["{filepath}"]\n\n' +

            '[config.builder_config]\n' +
            'builder_type = "pyinstaller"\n' +
            f'entry_file = "{DATA_FOLDER}/{specfile}"'
        )

        ver = (botdata['settings']['run_command'].split(" "))[-1]
        print(ver)
        policy_file = VERSIONS[ver]().policy_path

        with open(f"{DATA_FOLDER}/{runfile}", 'w', encoding="utf-8") as run:
            run.write(
                "from emerald.src.bot import run\n" +
                f"run('{ver}')\n"
            )

        with open(f"{DATA_FOLDER}/{specfile}", 'w', encoding="utf-8") as spec:
            spec.write(
                "# -*- mode: python ; coding: utf-8 -*-\n\n" +

                "a = Analysis(" +
                    f"['{runfile}']," +
                    "pathex=['.']," +
                    "binaries=None," +
                    f"datas=[('../emerald/policy/{policy_file}', 'policy')]," +
                    "hiddenimports=[]," +
                    "hooksconfig={}," +
                    "runtime_hooks=[]," +
                    "excludes=[]," +
                    "noarchive=False," +
                    "optimize=0" +
                ")\n" +
                
                "pyz = PYZ(a.pure)\n"

                "exe = EXE(" +
                    "pyz," +
                    "a.scripts," +
                    "a.binaries," +
                    "a.datas," +
                    "[]," +
                    f"name='{id}'," +
                    "debug=False," +
                    "bootloader_ignore_signals=False," +
                    "strip=False," +
                    "upx=True," +
                    "upx_exclude=[]," +
                    "runtime_tmpdir=None," +
                    "console=True," +
                    "disable_windowed_traceback=False," +
                    "argv_emulation=False," +
                    "target_arch=None," +
                    "codesign_identity=None," +
                    "entitlements_file=None" +
                ")"
            )
    
