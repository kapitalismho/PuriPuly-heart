from __future__ import annotations

from puripuly_heart.core.osc.receiver import VrcMicState, VrcOscReceiver


def test_receiver_routes_controls_avatar_change_and_mute_on_one_socket() -> None:
    controls: list[tuple[str, tuple[object, ...]]] = []
    packets: list[tuple[str, tuple[object, ...]]] = []
    avatars: list[tuple[object, ...]] = []
    mutes: list[bool] = []
    receiver = VrcOscReceiver(
        state=VrcMicState(),
        mute_packet_handler=mutes.append,
        control_packet_handler=lambda address, values: controls.append((address, values)),
        avatar_change_handler=avatars.append,
        packet_handler=lambda address, values: packets.append((address, values)),
    )

    receiver.message_handler("/avatar/parameters/PuriPuly_Talk", True)
    receiver.message_handler("/avatar/change", "avatar-id")
    receiver.message_handler("/avatar/parameters/MuteSelf", True)

    assert controls == [("/avatar/parameters/PuriPuly_Talk", (True,))]
    assert packets == [
        ("/avatar/parameters/PuriPuly_Talk", (True,)),
        ("/avatar/change", ("avatar-id",)),
    ]
    assert avatars == [("avatar-id",)]
    assert mutes == [True]
