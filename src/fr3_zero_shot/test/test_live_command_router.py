from fr3_zero_shot.live_command_router import resolve_command


def test_visible_blue_pick_routes_to_live_pick():
    resolved = resolve_command("pick up the blue cube")

    assert resolved is not None
    assert resolved.module == "fr3_zero_shot.live_pick"
    assert resolved.intent == "pick"
    assert "pick up the blue cube" in resolved.args
    assert "--execute" in resolved.args


def test_blue_cup_routes_to_pick_place():
    resolved = resolve_command("pick up the blue cube and place it in the cup")

    assert resolved is not None
    assert resolved.module == "fr3_zero_shot.live_pick_place"
    assert resolved.intent == "pick_place"
    assert resolved.target_object == "cup"


def test_multi_cube_cup_routes_in_spoken_order():
    resolved = resolve_command("put the large red cube, blue cube, and yellow cubes all in the cup")

    assert resolved is not None
    assert resolved.module == "fr3_zero_shot.live_pick_place_sequence"
    assert resolved.intent == "multi_cube_cup_sequence"
    colors_index = resolved.args.index("--colors")
    assert resolved.args[colors_index + 1 : colors_index + 4] == ("red", "blue", "yellow")


def test_stack_routes_to_stack_sources():
    resolved = resolve_command("stack the yellow on the blue and then the small red on the yellow")

    assert resolved is not None
    assert resolved.module == "fr3_zero_shot.live_stack_cubes"
    assert resolved.intent == "stack_cubes"
    source_index = resolved.args.index("--stack-sources")
    assert resolved.args[source_index + 1 : source_index + 3] == ("yellow", "small_red")


def test_green_occluded_pick_routes_to_llm_strategy():
    resolved = resolve_command("pick up the partially occluded green cube without touching the cup")

    assert resolved is not None
    assert resolved.module == "fr3_zero_shot.live_green_occluded_pick"
    assert resolved.intent == "green_occluded_pick"
    assert "--llm-strategy" in resolved.args


def test_unsupported_command_returns_none():
    assert resolve_command("wave hello") is None
