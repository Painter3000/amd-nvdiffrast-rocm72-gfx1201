from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from util.func import load_obj, make_sphere,make_star_cameras, normalize_vertices, save_obj, save_images
from util.render import NormalsRenderer
from tqdm import tqdm
from util.snapshot import snapshot
try:
    from util.view import show
except:
    show = None


def save_alpha_images(alpha, out_dir):
    """Save single-channel alpha tensors through save_images().

    util.func.save_images expects RGB-like tensors and current imageio/Pillow
    rejects [H,W,1] images. Repeat alpha to RGB for preview/debug output.
    This does not affect training or loss values.
    """
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        alpha = alpha.repeat(1, 1, 1, 3)
    save_images(alpha, out_dir)

fname = 'data/lucy.obj'
steps = 100
snapshot_step = 1

mv,proj = make_star_cameras(4,4)
renderer = NormalsRenderer(mv,proj,[512,512])

target_vertices,target_faces =  load_obj(fname)
target_vertices = normalize_vertices(target_vertices)
target_normals = calc_vertex_normals(target_vertices,target_faces)
target_images = renderer.render(target_vertices,target_normals,target_faces)
save_images(target_images[...,:3], './out/target_images/')
save_alpha_images(target_images[...,3:], './out/target_alpha/')

vertices,faces = make_sphere(level=2,radius=.5)

opt = MeshOptimizer(vertices,faces)
vertices = opt.vertices
snapshots = []

for i in tqdm(range(steps)):
    opt.zero_grad()
    normals = calc_vertex_normals(vertices,faces)
    images = renderer.render(vertices,normals,faces)
    loss = (images-target_images).abs().mean()
    loss.backward()
    opt.step()

    if show and i%snapshot_step==0:
        snapshots.append(snapshot(opt))

    vertices,faces = opt.remesh()

print(f"FINAL_LOSS={loss.item():.8f}")
print(f"FINAL_VERTICES={vertices.shape[0]}")
print(f"FINAL_FACES={faces.shape[0]}")

save_obj(vertices,faces,'./out/result.obj')
save_images(images[...,:3], './out/images/')
save_alpha_images(images[...,3:], './out/alpha/')

if show:
    show(target_vertices,target_faces,snapshots)
