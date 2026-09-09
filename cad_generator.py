# cad_generator.py
# Generates PDC drill bit components (body, blades, cutters, nozzles, gauge pads, junk slots)
# based on IADC parameters and predicted wear rate.

import math
import cadquery as cq

def make_pdc_cutter(radius, height, dome_factor=1.0):
    """Create a single domed PDC cutter (cylinder + spherical dome)."""
    base = cq.Solid.makeCylinder(radius, height)
    dome_radius = radius * 2.0
    sphere = cq.Solid.makeSphere(dome_radius)
    sphere = sphere.translate((0, 0, height - dome_radius + radius * dome_factor))
    return base.fuse(sphere)

def generate_pdc_components(iadc_params, wear_rate):
    """
    Generate all components of a PDC drill bit.
    Returns a dict with keys: 'body', 'blades', 'cutters', 'nozzles'.
    """
    wear_rate = max(0.001, wear_rate)
    scale = 1.0 + 0.1 * wear_rate

    # IADC parameters
    cutter_size = iadc_params['cutter']['size_mm']
    cutter_count_range = iadc_params['cutter']['count_range']
    num_cutters = int(cutter_count_range[0] + (cutter_count_range[1] - cutter_count_range[0]) * 0.5)
    profile_key = iadc_params['profile']['key']

    # Slot dimensions depend on profile
    slot_depth_factor = {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8}[profile_key]
    slot_width_factor = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0}[profile_key]

    # Base dimensions (all scaled by wear rate)
    bit_radius = max(25.0, 42.5 * scale)
    shank_radius = max(20.0, 25.0 * scale)
    gauge_length = 15.0 * scale
    face_height = 10.0 * scale
    shank_length = 60.0 * scale
    cutter_radius = cutter_size / 2.0
    cutter_height = 45.0 * scale          # optimal visibility
    blade_height = 63.0 * scale           # just below cutters
    blade_width = 6.0 * scale
    blade_start_radius = 10.0 * scale

    # Nozzle parameters – made tall enough to be clearly visible
    nozzle_radius = bit_radius * 0.10
    nozzle_height = 61.0 * scale          # final height for visibility

    slot_depth = face_height * slot_depth_factor
    slot_width = blade_width * slot_width_factor

    # --- Shank ---
    shank = cq.Workplane("XY").circle(shank_radius).extrude(shank_length)
    shank = shank.faces("<Z").edges().chamfer(5.0 * scale)

    # --- Main body ---
    top_z = shank_length + gauge_length + face_height
    body = (
        cq.Workplane("XY")
        .workplane(offset=shank_length)
        .circle(shank_radius)
        .workplane(offset=top_z)
        .circle(bit_radius)
        .loft(combine=True)
    )
    body = shank.union(body)

    # --- Segmented gauge pads (behind each blade) ---
    num_blades = 3
    pads = cq.Workplane()
    for i in range(num_blades):
        base_angle = i * 360.0 / num_blades
        pad_radial = 4.0 * scale
        pad_tangential = 10.0 * scale
        pad_height = 4.0 * scale
        pad = (
            cq.Workplane("XY")
            .box(pad_tangential, pad_radial, pad_height)
            .translate((bit_radius - pad_radial/2, 0, shank_length + gauge_length - 2.0 * scale))
            .rotate((0, 0, 0), (0, 0, 1), base_angle)
        )
        pads = pads.union(pad)
    if pads and pads.objects:
        body = body.union(pads)

    # --- Blades (straight radial) ---
    blades = cq.Workplane()
    for i in range(num_blades):
        base_angle = i * 360.0 / num_blades
        wp = cq.Workplane("XY").workplane(offset=top_z)
        wp = wp.transformed(rotate=(0, 0, base_angle))
        wp = wp.moveTo(blade_start_radius, -blade_width/2.0)
        blade_length = bit_radius - blade_start_radius - 2.0 * scale
        blade = wp.rect(blade_length, blade_width).extrude(blade_height, combine=False)
        blades = blades.union(blade)
    if blades and blades.objects:
        body = body.union(blades)

    # --- Cutters (no exclusion, placed along blades) ---
    cutters_list = []
    cutters_per_blade = max(3, num_cutters // num_blades)
    for i in range(num_blades):
        base_angle = i * 360.0 / num_blades
        for j in range(cutters_per_blade):
            t = (j + 0.5) / cutters_per_blade
            r = blade_start_radius + 2.0 + t * (bit_radius - 15.0 - blade_start_radius - 4.0 - cutter_radius)
            x = r * math.cos(math.radians(base_angle))
            y = r * math.sin(math.radians(base_angle))
            cutter_z = top_z + cutter_height / 2
            cutter = make_pdc_cutter(cutter_radius, cutter_height, dome_factor=1.0)
            cutter = cutter.translate((x, y, cutter_z))
            cutters_list.append(cutter)

    if cutters_list:
        cutters = cq.Workplane(obj=cq.Compound.makeCompound(cutters_list))
    else:
        cutters = cq.Workplane()

    # --- Central flushing nozzle (cut hole) ---
    central_nozzle = cq.Solid.makeCylinder(bit_radius * 0.12, face_height * 0.8)
    central_nozzle = central_nozzle.translate((0, 0, top_z - face_height * 0.4))
    body = body.cut(central_nozzle)

    # --- Additional nozzles (holes + visible protrusions) ---
    nozzles_list = []
    for i in range(num_blades):
        angle = (i * 360.0 / num_blades) + 60.0
        r = bit_radius * 0.35
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))

        # Cut hole
        hole = cq.Solid.makeCylinder(nozzle_radius, face_height * 0.8)
        hole = hole.translate((x, y, top_z - face_height * 0.4))
        body = body.cut(hole)

        # Add tall protrusion (nozzle)
        nozzle_protrusion = cq.Solid.makeCylinder(nozzle_radius, nozzle_height)
        nozzle_protrusion = nozzle_protrusion.translate((x, y, top_z))
        nozzles_list.append(nozzle_protrusion)

    if nozzles_list:
        nozzles = cq.Workplane(obj=cq.Compound.makeCompound(nozzles_list))
    else:
        nozzles = cq.Workplane()

    # --- Junk slots (triangular grooves between blades) ---
    slots = cq.Workplane()
    for i in range(num_blades):
        base_angle = i * 360.0 / num_blades + 30.0
        wp = cq.Workplane("XY").workplane(offset=top_z)
        wp = wp.transformed(rotate=(0, 0, base_angle))
        inner_r = blade_start_radius * 1.2
        outer_r = bit_radius - 2.0 * scale
        half_width = slot_width / 2.0
        pts = [(inner_r, -half_width), (inner_r, half_width), (outer_r, 0)]
        tri = wp.polyline(pts).close()
        slot = tri.extrude(-slot_depth, combine=False)
        slots = slots.union(slot)
    if slots and slots.objects:
        body = body.cut(slots)

    # Optional overlap warning (does not stop execution)
    # (cutter‑nozzle overlap check is omitted for simplicity)

    return {'body': body, 'blades': blades, 'cutters': cutters, 'nozzles': nozzles}