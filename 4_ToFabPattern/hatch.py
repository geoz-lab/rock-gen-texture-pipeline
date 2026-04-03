import ezdxf
from ezdxf.math import Vec2, area as polygon_area

# --- SETTINGS ---
input_dxf = 'rock_background.dxf'
output_dxf = 'rock_background_hatched.dxf'
hatch_layer = 'hatch'
min_hole_area = 10.0  # holes smaller than this will be ignored

# Specify a known point inside the background to hatch (X, Y)
seed_point = (550.0, 17950.0)

def point_in_polygon(point, polygon):
    x, y = point
    num = len(polygon)
    j = num - 1
    inside = False
    for i in range(num):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

# --- STEP 1: Load DXF ---
print("Loading DXF...")
doc = ezdxf.readfile(input_dxf)
msp = doc.modelspace()

# --- STEP 2: Collect polylines ---
print("Collecting polylines...")
outer_polyline = None
hole_polylines = []

for e in msp:
    if e.dxftype() == 'LWPOLYLINE' and e.is_closed:
        points = [Vec2(p[0], p[1]) for p in e.get_points()]
        poly_area = abs(polygon_area(points))
        if poly_area < min_hole_area:
            continue  # skip tiny particles

        if point_in_polygon(seed_point, points):
            if outer_polyline is None or poly_area > abs(polygon_area([Vec2(p[0], p[1]) for p in outer_polyline.get_points()])):
                outer_polyline = e
        else:
            hole_polylines.append((poly_area, e))

if outer_polyline is None:
    raise ValueError("No outer boundary found containing the seed point!")

print(f"Found outer boundary with area {abs(polygon_area([Vec2(p[0], p[1]) for p in outer_polyline.get_points()])):.2f}")
print(f"Found {len(hole_polylines)} hole candidate(s)")

# --- STEP 3: Prepare hatch ---
hatch = msp.add_hatch(color=7)  # color 7 = black/white depending on background
hatch.dxf.layer = hatch_layer

# Add outer boundary first
outer_points = [Vec2(p[0], p[1]) for p in outer_polyline.get_points()]
hatch.paths.add_polyline_path(outer_points, is_closed=True)

# Add holes (no is_outer argument, just order matters)
for poly_area, e in hole_polylines:
    if poly_area < min_hole_area:
        continue
    hole_points = [Vec2(p[0], p[1]) for p in e.get_points()]
    hatch.paths.add_polyline_path(hole_points, is_closed=True)

# Set hatch pattern to SOLID
hatch.set_pattern_fill('SOLID')

print("Hatch created on layer 'hatch'.")

# --- STEP 4: Save DXF ---
doc.saveas(output_dxf)
print(f"DXF saved as: {output_dxf}")
