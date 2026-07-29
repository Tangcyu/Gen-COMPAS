import os

from utils.mdtraj_io import filter_dcdplugin_messages


def test_filter_dcdplugin_messages_hides_only_known_lines(capfd):
    with filter_dcdplugin_messages():
        os.write(
            1,
            b"dcdplugin) detected standard 32-bit DCD file of native endianness\n",
        )
        os.write(
            2,
            b"dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)\n",
        )
        os.write(1, b"dcdplugin) retained diagnostic\n")

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "detected standard 32-bit DCD" not in combined
    assert "CHARMM format DCD file" not in combined
    assert "dcdplugin) retained diagnostic" in combined
