"""bg_far_city.png — дальний слой параллакса (1800x600), бесшовный по горизонтали. Процедурная отрисовка."""
import numpy as np, cv2, math
from PIL import Image, ImageDraw

W, H, S = 1800, 600, 3
w, h = W*S, H*S
rng = np.random.default_rng(2026)

def lerp(a, b, t): return a + (b - a) * t
def smooth(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t*t*(3-2*t)

def blur_wrap(img, sigma):
    """Гауссово размытие с циклической обёрткой по X (для бесшовности)."""
    pad = int(sigma*4)+2
    p = np.pad(img, ((0,0),(pad,pad)) + ((0,0),)*(img.ndim-2), mode='wrap')
    p = cv2.GaussianBlur(p, (0,0), sigma)
    return p[:, pad:-pad]

# ---------------------------------------------------------------- небо
Y = (np.arange(h)/h)[:,None]
stops = [(0.00,(110,126,156)),(0.22,(136,138,176)),(0.36,(172,150,180)),(0.45,(226,166,150)),
         (0.51,(247,192,130)),(0.56,(253,213,148)),(0.62,(244,202,160)),(0.78,(196,164,164)),(1.00,(132,120,142))]
ys = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops], np.float32)
col = np.stack([np.interp(Y[:,0], ys, cs[:,i]) for i in range(3)], axis=1)      # (h,3)
sky = np.repeat(col[:,None,:], w, axis=1).astype(np.float32)

# солнце: чуть левее центра, у горизонта
sx, sy, sr = 0.43*w, 0.505*h, 62*S
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
d = np.hypot(xx-sx, (yy-sy)*1.0)
glow = np.exp(-(d/(330*S))**1.6)[...,None]
sky = sky*(1-0.55*glow) + np.array([255,222,150],np.float32)*0.55*glow
glow2 = np.exp(-(d/(130*S))**1.4)[...,None]
sky = sky*(1-0.55*glow2) + np.array([255,232,165],np.float32)*0.55*glow2
disk = smooth(sr+2*S, sr-2*S, d)[...,None]
sky = sky*(1-disk) + np.array([255,238,176],np.float32)*disk
# горизонтальное тёплое свечение у линии горизонта
hor = np.exp(-((yy-0.545*h)/(46*S))**2)[...,None] * np.exp(-((xx-sx)/(520*S))**2)[...,None]
sky = sky*(1-0.35*hor) + np.array([255,214,150],np.float32)*0.35*hor

# ---------------------------------------------------------------- облака
def cloud(cx, cy, sxs, sys_, n, seed, warm=0.6, alpha=0.9):
    r = np.random.default_rng(seed)
    m = Image.new('L', (w, h), 0); dm = ImageDraw.Draw(m)
    for k in (-1, 0, 1):
        for _ in range(n):
            ex = cx + k*w + r.normal(0, sxs*0.38)*S
            ey = cy + r.normal(0, sys_*0.25)*S
            rx = r.uniform(0.18, 0.42)*sxs*S; ry = r.uniform(0.35, 0.8)*sys_*S
            dm.ellipse([ex-rx, ey-ry, ex+rx, ey+ry], fill=255)
    a = np.array(m, np.float32)/255
    a = blur_wrap(a, 2.2*S)
    a = smooth(0.42, 0.62, a)
    # затенение: холодная верхушка, тёплое «подбрюшье», ближе к солнцу — теплее
    top = cy - sys_*S*0.9
    t = np.clip((yy - top)/(sys_*S*1.8), 0, 1)[...,None]
    cool = np.array([172,160,196],np.float32); hot = np.array([250,188,150],np.float32)
    near_sun = np.exp(-((xx - sx)/(620*S))**2)[...,None]
    c = cool*(1-t) + hot*t
    c = c*(1-warm*near_sun*0.6) + np.array([255,206,150],np.float32)*warm*near_sun*0.6
    # блик на верхней кромке
    edge = np.clip(a - np.roll(a, 3*S, axis=0), 0, 1)[...,None]
    c = c + edge*np.array([40,34,20],np.float32)
    return c, (a*alpha)[...,None]

layers_c = [
    (0.14*w, 0.30*h/S, 360, 34, 11, 11, 0.5, 0.85),
    (0.62*w, 0.22*h/S, 420, 30, 12, 12, 0.5, 0.80),
    (0.86*w, 0.36*h/S, 300, 26, 9, 13, 0.7, 0.85),
    (0.30*w, 0.43*h/S, 260, 16, 8, 14, 0.9, 0.80),
    (0.56*w, 0.40*h/S, 230, 14, 7, 15, 0.9, 0.75),
    (0.97*w, 0.47*h/S, 330, 14, 9, 16, 0.9, 0.75),
    (0.04*w, 0.12*h/S, 240, 16, 7, 17, 0.2, 0.55),
]
for cx, cy, sxs, sys_, n, seed, warm, al in layers_c:
    c, a = cloud(cx, cy*S, sxs, sys_, n, seed, warm, al)
    sky = sky*(1-a) + c*a

# мягкая «кистевая» текстура неба: горизонтальные мазки (периодичные по X)
noise = rng.normal(0, 1, (h//3, w//3)).astype(np.float32)
k = np.zeros((1, 61), np.float32); k[0, :] = 1/61
pad = k.shape[1] // 2
noise_padded = np.pad(noise, ((0, 0), (pad, pad)), mode='wrap')
streak = cv2.filter2D(noise_padded, -1, k, borderType=cv2.BORDER_REPLICATE)[:, pad:-pad]
streak = cv2.resize(streak, (w, h), interpolation=cv2.INTER_CUBIC)
streak = blur_wrap(streak, 3*S)
sky += (streak/ (np.abs(streak).max()+1e-6) * 5.0)[...,None]

# ---------------------------------------------------------------- силуэты города
def poly(d, pts, fill): d.polygon([(x*S, y*S) for x, y in pts], fill=fill)

def building(d, x, base, bw, bh, kind, r, fill, windows=False):
    top = base - bh
    if kind == 'flat':
        poly(d, [(x,base),(x,top),(x+bw,top),(x+bw,base)], fill)
    elif kind == 'stepped':
        poly(d, [(x,base),(x,top),(x+bw,top),(x+bw,base)], fill)
        sw = bw*r.uniform(0.45, 0.7); sx_ = x + (bw-sw)/2
        poly(d, [(sx_,top),(sx_,top-bh*0.14),(sx_+sw,top-bh*0.14),(sx_+sw,top)], fill)
        if r.random() < 0.6: poly(d, [(x+bw/2-0.8,top-bh*0.14),(x+bw/2-0.8,top-bh*0.14-r.uniform(14,34)),(x+bw/2+0.8,top-bh*0.14-r.uniform(14,34)),(x+bw/2+0.8,top-bh*0.14)], fill)
    elif kind == 'slant':
        cut = r.uniform(0.55, 0.85)
        poly(d, [(x,base),(x,top),(x+bw,top+bh*(1-cut)),(x+bw,base)], fill)
    elif kind == 'collapsed':
        pts = [(x, base), (x, top)]; n = r.integers(4, 8)
        for i in range(1, n+1):
            px = x + bw*i/n; py = top + bh*r.uniform(0.0, 0.32)*(1 if i % 2 else 1.7)
            pts.append((px, py))
        pts += [(x+bw, base)]
        poly(d, pts, fill)
    elif kind == 'spire':
        poly(d, [(x,base),(x,top),(x+bw,top),(x+bw,base)], fill)
        poly(d, [(x+bw*0.2,top),(x+bw/2,top-bh*r.uniform(0.18,0.3)),(x+bw*0.8,top)], fill)
    elif kind == 'twin':
        w1 = bw*0.45; h2 = bh*r.uniform(0.6, 0.9)
        poly(d, [(x,base),(x,top),(x+w1,top),(x+w1,base)], fill)
        poly(d, [(x+bw-w1,base),(x+bw-w1,base-h2),(x+bw,base-h2),(x+bw,base)], fill)
        poly(d, [(x+w1,top+bh*0.35),(x+w1,top+bh*0.42),(x+bw-w1,base-h2+bh*0.35),(x+bw-w1,base-h2+bh*0.42)], fill)
    if windows and bw > 22:
        for _ in range(int(r.integers(2, 7))):
            wx = x + r.uniform(3, bw-5); wy = top + r.uniform(8, bh*0.8)
            d.rectangle([wx*S, wy*S, (wx+2.2)*S, (wy+3)*S], fill=(255, 205, 135, 255))

def chimneys(d, x, base, r, fill):
    poly(d, [(x,base),(x,base-26),(x+70,base-26),(x+70,base)], fill)
    for i in range(int(r.integers(2, 4))):
        cx = x + 8 + i*20; ch = r.uniform(54, 100)
        poly(d, [(cx,base-26),(cx+1.2,base-26-ch),(cx-3,base-26-ch),(cx-3,base-26-ch-5),(cx+9,base-26-ch-5),(cx+9,base-26-ch),(cx+5.2,base-26-ch),(cx+6.4,base-26)], fill)

def cooling(d, x, base, bw, bh, fill):
    poly(d, [(x,base),(x+bw*0.18,base-bh*0.5),(x+bw*0.2,base-bh),(x+bw*0.8,base-bh),(x+bw*0.82,base-bh*0.5),(x+bw,base)], fill)

def mast(d, x, base, mh, fill):
    poly(d, [(x-1.2,base),(x-0.6,base-mh),(x+0.6,base-mh),(x+1.2,base)], fill)
    for i in range(1, 6):
        yv = base - mh*i/6; ww = 7*(1-i/6)+1.2
        d.line([((x-ww)*S, yv*S), ((x+ww)*S, (yv)*S)], fill=fill, width=int(1.2*S))

def broken_arch(d, cx, base, rad, fill):
    th = 9
    d.arc([(cx-rad)*S, (base-rad*2-30)*S, (cx+rad)*S, (base-30)*S], 205, 345, fill=fill, width=th*S)
    d.arc([(cx-rad)*S, (base-rad*2-30)*S, (cx+rad)*S, (base-30)*S], 20, 130, fill=fill, width=th*S)
    for sgn in (-1, 1):
        px = cx + sgn*rad
        poly(d, [(px-6,base),(px-5,base-rad-30),(px+5,base-rad-30),(px+6,base)], fill)

def layer(seed, color, n_target, hmin, hmax, wmin, wmax, base, kinds, special=(), windows=False, center_scale=0.38, jitter_base=10):
    r = np.random.default_rng(seed)
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    x = -20.0
    while x < W + 20:
        bw = r.uniform(wmin, wmax); bh = r.uniform(hmin, hmax)
        mid = x/W
        if 0.37 < mid < 0.62: bh *= center_scale        # центр у солнца — спокойный, низкий
        kind = r.choice(kinds)
        var = r.integers(-9, 10)
        fill = (int(color[0]+var), int(color[1]+var), int(color[2]+var), 255)
        b = base + r.uniform(-jitter_base, jitter_base)
        for k in (-1, 0, 1):
            building(d, x + k*W, b + 70, bw, bh + 70, kind, np.random.default_rng(int(r.integers(1e9))), fill, windows)
        x += bw * r.uniform(0.55, 1.05)
    for sp in special:
        for k in (-1, 0, 1):
            t = sp[0]
            if t == 'chimneys': chimneys(d, sp[1]+k*W, base+70, r, (color[0],color[1],color[2],255))
            if t == 'cooling':  cooling(d, sp[1]+k*W, base+70, sp[2], sp[3], (color[0],color[1],color[2],255))
            if t == 'mast':     mast(d, sp[1]+k*W, base+70-sp[3]*0, sp[2], (color[0],color[1],color[2],255))
            if t == 'arch':     broken_arch(d, sp[1]+k*W, base+70, sp[2], (color[0],color[1],color[2],255))
    return np.array(img, np.float32)

def composite(sky, L, haze0, shade, rim=None):
    a = (L[...,3]/255)[...,None]
    a = blur_wrap(a[...,0], 0.9*S)[...,None]
    base_rgb = L[...,:3]
    yv = (np.arange(h)/h)[:,None,None]
    hz = haze0 + (1-haze0)*smooth(0.36, 0.64, yv)                 # к горизонту здания растворяются в дымке
    hz = np.broadcast_to(hz, (h, w, 1))
    # вертикальный градиент внутри здания: светлее к низу (тёплый отсвет)
    lit = base_rgb*(1-0.10*smooth(0.30, 0.55, yv))
    out_rgb = lit*(1-hz) + sky*hz
    if rim is not None:                                            # тёплая кромка со стороны солнца
        m = (a[...,0] > 0.5).astype(np.float32)
        side = np.where(xx > sx, 1, -1)
        shifted = np.where(side > 0, np.roll(m, 3*S, axis=1), np.roll(m, -3*S, axis=1))
        edge = np.clip(m - shifted, 0, 1)[...,None]
        out_rgb = out_rgb + edge*np.array(rim, np.float32)*(1-hz)
    res = sky*(1-a*shade) + out_rgb*a*shade
    return res

far  = layer(11, (176,164,194), 0, 55, 150, 12, 34, 396, ['flat','stepped','slant','spire','collapsed'],
             special=[('mast', 0.20*W, 120, 0), ('mast', 0.83*W, 150, 0), ('cooling', 0.07*W, 40, 70)])
mid  = layer(22, (150,138,170), 0, 80, 210, 16, 52, 406, ['flat','stepped','slant','collapsed','twin','spire'],
             special=[('chimneys', 0.12*W), ('cooling', 0.93*W, 54, 90), ('mast', 0.70*W, 130, 0)])
near = layer(33, (124,112,146), 0, 70, 250, 20, 66, 416, ['flat','stepped','slant','collapsed','twin'],
             special=[('arch', 0.74*W, 62), ('mast', 0.30*W, 170, 0), ('chimneys', 0.88*W)], windows=True)

sky = composite(sky, far,  0.66, 0.95)
sky = composite(sky, mid,  0.50, 0.97, rim=(26, 20, 6))
sky = composite(sky, near, 0.34, 1.00, rim=(52, 38, 10))

# дымка у горизонта: растянутые тёплые полосы
for i in range(9):
    cx = rng.uniform(0, w); cy = rng.uniform(0.54, 0.64)*h
    m = Image.new('L', (w, h), 0); dm = ImageDraw.Draw(m)
    for k in (-1, 0, 1):
        rx = rng.uniform(160, 320)*S; ry = rng.uniform(10, 20)*S
        dm.ellipse([cx+k*w-rx, cy-ry, cx+k*w+rx, cy+ry], fill=255)
    a = blur_wrap(np.array(m, np.float32)/255, 14*S)[...,None]
    sky = sky*(1-0.32*a) + np.array([252,208,160],np.float32)*0.32*a
# общий туман ниже горизонта
fog = smooth(0.56, 0.72, np.arange(h)[:,None,None]/h)
sky = sky*(1-0.55*fog) + np.array([238,198,160],np.float32)*0.55*fog*(1 - 0.35*smooth(0.72, 1.0, np.arange(h)[:,None,None]/h)) \
      + np.array([150,130,150],np.float32)*0.55*fog*0.35*smooth(0.72, 1.0, np.arange(h)[:,None,None]/h)

# мелкое зерно + низкочастотная неоднородность (периодичны по X)
grain = rng.normal(0, 1, (h, w)).astype(np.float32)
sky += (blur_wrap(grain, 0.8*S)*2.2)[...,None]
lowf = blur_wrap(rng.normal(0, 1, (h//12, w//12)).astype(np.float32), 5)
lowf = cv2.resize(lowf, (w, h), interpolation=cv2.INTER_CUBIC)
sky *= (1 + 0.025*lowf/ (np.abs(lowf).max()+1e-6))[...,None]

out = cv2.resize(np.clip(sky, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
Image.fromarray(out).save('bg_far_city.png')
print('saved', out.shape)