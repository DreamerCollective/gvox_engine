#pragma once

b32 custom_brush_should_edit(in BrushInput brush) {
    return length(brush.p) < PLAYER.edit_radius;
}

Voxel custom_brush_kernel(in BrushInput brush) {
    return Voxel(INPUT.settings.brush_color, BlockID_Stone);
}
