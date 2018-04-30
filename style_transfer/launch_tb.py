import webbrowser
from subprocess import call

webbrowser.open_new_tab("http://DESKTOP-28QRJQN:6006")
call(["tensorboard", "--logdir=tensorboard"])