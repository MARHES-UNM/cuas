#!/bin/bash
Xvfb -screen 0 1024x768x24 -ac &
sleep 5
export DISPLAY=:0.0
x11vnc -rfbauth ~/.vnc/passwd -noxrecord -noxfixes -noxdamage -forever -display :0 -bg
xfce4-session &

[ -f "/app.sh" ] && sh /app.sh || /bin/bash