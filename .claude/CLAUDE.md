# Shell commands

Never write an `rm` whose path comes from a shell variable (for example `rm -f $S/*.png` or `rm -rf $DIR`). Claude Code flags it as a
dangerous delete and stops for the owner to click "Allow once". Instead:
- use the literal absolute path: `rm -f /private/tmp/.../film5/st/*.png`
- or guard the variable: `rm -f "${S:?}"/*.png`
- or delete with find on a literal directory: `find /private/tmp/.../film5/st -name '*.png' -delete`
