from trainer import config

config.is_test = True
config.epochs = 1

def test_graybox_e2e():
    from trainer import graybox_task

def test_blackbox_e2e():
    from trainer import blackbox_task
