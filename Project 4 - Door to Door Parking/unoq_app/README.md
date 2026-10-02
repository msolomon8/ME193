# UNO Q app (Arduino App Lab)

- `sketch/sketch.ino` - MCU side: Maker Drive motors (M1A=D10, M1B=D9, M2A=D6, M2B=D5) + LED matrix dot, exposed over Bridge as `drive(m1, m2)` and `dot(col, row)`. Watchdog stops motors after 1 s without commands.
- `python/main.py` - Linux side: subscribes to `ME193/minifig`, shows the minifig position on the matrix, drives forward/backward until it is centered.
- `python/requirements.txt` - `paho-mqtt`

This folder is a copy for GitHub; the app is run from App Lab.
