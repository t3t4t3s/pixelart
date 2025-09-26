from PIL import Image, ImageFilter, ImageOps, ImageChops
import random
from typing import Tuple

def resize_match_ref_height_and_crop(base: Image.Image,
                                     ref: Image.Image,
                                     offset_x: int = 0,
                                     offset_y: int = 0,
                                     resample=Image.Resampling.BICUBIC,
                                     pad_if_narrow: bool = True,
                                     pad_fill=(0, 0, 0, 0)) -> Image.Image:
    """
    - Redimensionne `base` pour que sa hauteur == ref.height (ratio conservé).
    - Aligne `base` à la taille de `ref` en:
        * rognant (si base_redim.width >= ref.width)
        * sinon, en ajoutant des marges (si base_redim.width < ref.width et pad_if_narrow=True)
    - offset_x / offset_y décalent la fenêtre de recadrage (ou le placement si padding).
    """
    ref_w, ref_h = ref.size
    b_w, b_h = base.size

    # 1) Mise à l’échelle sur la hauteur (aspect préservé)
    scale = ref_h / b_h
    new_w = max(1, int(round(b_w * scale)))
    base_scaled = base.resize((new_w, ref_h), resample=resample)

    # 2) Ajustement en largeur
    if new_w >= ref_w:
        # On CROPE dans base_scaled → fenêtre (left, top, right, bottom)
        # fenêtre centrée, puis on applique le décalage demandé
        left = (new_w - ref_w) // 2 + offset_x
        top  = 0 + offset_y  # en général 0 car on a déjà la bonne hauteur
        # Clamp pour rester dans l'image
        left = max(0, min(left, new_w - ref_w))
        top  = max(0, min(top,  ref_h - ref_h))  # donc 0
        box = (left, top, left + ref_w, top + ref_h)
        return base_scaled.crop(box)
    else:
        # Trop étroit : on PAD pour atteindre ref.size (letterbox en largeur)
        if not pad_if_narrow:
            # Optionnel : on peut choisir d’upscaler une seconde fois (peu recommandé)
            return base_scaled.resize(ref.size, resample=resample)

        # Créer un canevas ref.size et coller base_scaled avec décalage
        mode = "RGBA" if base_scaled.mode == "RGBA" else "RGB"
        canvas = Image.new(mode, (ref_w, ref_h), pad_fill)
        # position horizontale centrée + offset_x
        x = (ref_w - new_w) // 2 + offset_x
        y = 0 + offset_y
        # clamp pour éviter de sortir du cadre
        x = max(-(new_w - ref_w), min(x, ref_w - 1))
        y = max(-(ref_h - ref_h), min(y, ref_h - 1))
        canvas.paste(base_scaled, (x, y))
        return canvas
    
def upscale_preserve_aspect(
    img: Image.Image,
    *,
    scale: float | None = None,          # ex. 2.0 => x2
    width: int | None = None,            # fixe la largeur (hauteur auto)
    height: int | None = None,           # fixe la hauteur (largeur auto)
    longer: int | None = None,           # fixe la grande dimension
    shorter: int | None = None,          # fixe la petite dimension
    box: tuple[int, int] | None = None,  # (max_w, max_h), CONTAIN (seulement upscaling)
    megapixels: float | None = None,     # cible en MP (ex. 8.0)
    resample = Image.Resampling.LANCZOS, # meilleur pour agrandir
    sharpen: bool = True,                # renforce un peu après upscale
) -> Image.Image:
    """
    Retourne une VERSION AGRANDIE de img (jamais plus petite), en conservant le ratio.
    Indique exactement UN des paramètres d'objectif (scale, width, height, longer, shorter, box, megapixels).
    """
    W, H = img.size
    if sum(p is not None for p in (scale, width, height, longer, shorter, box, megapixels)) != 1:
        raise ValueError("Spécifie exactement un objectif parmi: scale/width/height/longer/shorter/box/megapixels")

    # Calcul de la taille cible (new_w, new_h) en respectant le ratio
    if scale is not None:
        if scale <= 1.0:
            # on ne réduit pas; si tu veux permettre la réduction, change cette règle
            return img.copy()
        new_w, new_h = int(round(W*scale)), int(round(H*scale))

    elif width is not None:
        if width <= W:
            return img.copy()
        new_w = width
        new_h = int(round(H * (width / W)))

    elif height is not None:
        if height <= H:
            return img.copy()
        new_h = height
        new_w = int(round(W * (height / H)))

    elif longer is not None:
        L = max(W, H)
        if longer <= L:
            return img.copy()
        s = longer / L
        new_w, new_h = int(round(W*s)), int(round(H*s))

    elif shorter is not None:
        S = min(W, H)
        if shorter <= S:
            return img.copy()
        s = shorter / S
        new_w, new_h = int(round(W*s)), int(round(H*s))

    elif box is not None:
        max_w, max_h = box
        # on ne FAIT QUE de l'upscaling: si l'image rentre déjà, on la laisse
        if W <= max_w and H <= max_h:
            return img.copy()
        # sinon, agrandir jusqu'à toucher une des limites (contain)
        s = min(max_w / W, max_h / H)
        new_w, new_h = int(round(W*s)), int(round(H*s))

    else:  # megapixels
        if megapixels <= 0:
            raise ValueError("megapixels doit être > 0")
        target_pixels = megapixels * 1_000_000
        cur_pixels = W * H
        if target_pixels <= cur_pixels:
            return img.copy()
        s = (target_pixels / cur_pixels) ** 0.5
        new_w, new_h = int(round(W*s)), int(round(H*s))

    # Redimensionnement
    out = img.resize((new_w, new_h), resample=resample)

    # Légère accentuation post-upscale pour compenser le lissage
    if sharpen:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    return out


def split_grid(img: Image.Image, rows=2, cols=3):
    """Retourne une liste de 6 sous-images (row-major)."""
    W, H = img.size
    # Bords “propres” par arrondi pour éviter les décalages cumulés
    x_edges = [round(W * c / cols) for c in range(cols + 1)]
    y_edges = [round(H * r / rows) for r in range(rows + 1)]

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left   = x_edges[c]
            right  = x_edges[c + 1]
            upper  = y_edges[r]
            lower  = y_edges[r + 1]
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles

def select_black_pixels(
    img: Image.Image,
    r_max: int = 20,
    g_max: int = 20,
    b_max: int = 20,
    mode: str = "mask",   # "mask" -> retourne un masque 'L' (0..255)
                          # "rgba" -> retourne une image RGBA : noirs conservés, reste transparent
) -> Image.Image:
    """
    Sélectionne les pixels noirs selon des seuils par canal:
      pixel sélectionné si (R <= r_max) ET (G <= g_max) ET (B <= b_max)

    Paramètres
    ----------
    img   : Image PIL d'entrée (n'importe quel mode; convertie en RGB pour le test)
    r_max, g_max, b_max : seuils par canal (0..255)
    mode  : "mask" -> Image 'L' (255= noir sélectionné, 0 = non-sélectionné)
            "rgba" -> Image 'RGBA' ne conservant que les pixels noirs (autres = alpha 0)

    Retour
    ------
    Image PIL (masque 'L' ou image 'RGBA' selon `mode`)
    """
    rgb = img.convert("RGB")
    R, G, B = rgb.split()

    # Seuils “<=” : on produit des masques 1-bit (0/255), puis on fait un AND logique
    r_mask_1 = R.point(lambda v: 255 if v <= r_max else 0, mode="1")
    g_mask_1 = G.point(lambda v: 255 if v <= g_max else 0, mode="1")
    b_mask_1 = B.point(lambda v: 255 if v <= b_max else 0, mode="1")

    # AND pixel-à-pixel (disponible sur mode "1")
    and_rg = ImageChops.logical_and(r_mask_1, g_mask_1)
    mask_1  = ImageChops.logical_and(and_rg, b_mask_1)  # 255 si (R<=r_max & G<=g_max & B<=b_max)

    if mode == "mask":
        # Convertir en 'L' (0..255) si tu préfères un masque 8 bits
        return mask_1.convert("L")

    elif mode == "rgba":
        # Conserver uniquement les pixels noirs : alpha = masque, RGB d'origine
        A = mask_1.convert("L")
        out = rgb.convert("RGBA")
        out.putalpha(A)  # autres pixels -> alpha 0
        return out

    else:
        raise ValueError("mode doit être 'mask' ou 'rgba'")
    

def subtract_constant_clip0(img: Image.Image, k: int = 200) -> Image.Image:
    """
    Soustrait k à chaque pixel et clippe à 0 (pas de valeurs négatives).
    - L : L' = max(L - k, 0)
    - RGB : par canal, alpha conservé si présent
    - RGBA/LA : on ne touche pas à l'alpha
    """
    k = max(0, min(255, int(k)))  # borne de sécurité
    lut_sub = [max(0, i - k) for i in range(256)]

    if img.mode == "L":
        return img.point(lut_sub)

    if img.mode == "RGB":
        r, g, b = img.split()
        r = r.point(lut_sub)
        g = g.point(lut_sub)
        b = b.point(lut_sub)
        return Image.merge("RGB", (r, g, b))

    if img.mode == "RGBA":
        r, g, b, a = img.split()
        r = r.point(lut_sub)
        g = g.point(lut_sub)
        b = b.point(lut_sub)
        return Image.merge("RGBA", (r, g, b, a))  # alpha inchangé

    if img.mode == "LA":
        l, a = img.split()
        l = l.point(lut_sub)
        return Image.merge("LA", (l, a))

    # Autres modes : convertis en RGB, applique, puis retourne en RGB
    base = img.convert("RGB")
    r, g, b = base.split()
    r = r.point(lut_sub); g = g.point(lut_sub); b = b.point(lut_sub)
    return Image.merge("RGB", (r, g, b))


def add_constant_clip255(img: Image.Image, k: int = 200) -> Image.Image:
    """
    Ajoute k à chaque pixel et clippe à 255.
    - L : L' = min(L + k, 255)
    - RGB : par canal ; alpha conservé si présent
    - RGBA / LA : on ne touche pas à l'alpha
    - Autres modes : converti en RGB d’abord
    """
    k = max(0, min(255, int(k)))  # borne de sécurité
    lut_add = [min(255, i + k) for i in range(256)]

    if img.mode == "L":
        return img.point(lut_add)

    if img.mode == "RGB":
        r, g, b = img.split()
        r = r.point(lut_add)
        g = g.point(lut_add)
        b = b.point(lut_add)
        return Image.merge("RGB", (r, g, b))

    if img.mode == "RGBA":
        r, g, b, a = img.split()
        r = r.point(lut_add)
        g = g.point(lut_add)
        b = b.point(lut_add)
        return Image.merge("RGBA", (r, g, b, a))  # alpha inchangé

    if img.mode == "LA":
        l, a = img.split()
        l = l.point(lut_add)
        return Image.merge("LA", (l, a))

    # Autres modes (ex. "P", "1", etc.) : passer par RGB
    base = img.convert("RGB")
    r, g, b = base.split()
    r = r.point(lut_add); g = g.point(lut_add); b = b.point(lut_add)
    return Image.merge("RGB", (r, g, b))


def subtract_images(
    img1: Image.Image,
    img2: Image.Image,
    match_size: bool = True,
    resample = Image.Resampling.BICUBIC,
    offset: int = 0,                           # décalage (>=0) après soustraction pour relever les noirs
    k = 210
) -> Image.Image:
    """
    - match_size : redimensionne img2 à la taille d'img1 si nécessaire
    - offset : ajoute un biais après subtraction pour éviter de tout clipper à 0
    """

    # Unifier tailles
    if match_size and (img2.size != img1.size):
        img2 = img2.resize(img1.size, resample=resample)

    # Choisir mode de travail (L ou RGB)
    work_mode = "RGB"
    im1 = img1.convert(work_mode)
    im2 = img2.convert(work_mode)

    # Enlever k à tous les pixels de im2
    im2 = subtract_constant_clip0(im2, k=k)
    #im2 = ImageOps.invert(im2)
    #im2 = add_constant_clip255(im2, k=20)

    # Soustraction : subtract(im1_scaled, im2, scale=1.0, offset=offset)
    out_rgb_or_l = ImageChops.subtract(im1, im2, offset=offset)
    #out_rgb_or_l = ImageChops.add(im1, im2, offset=offset)

    # Si img1 original était RGBA mais preserve_alpha=False, on renvoie RGB
    return out_rgb_or_l

def add_images(
    img1: Image.Image,
    img2: Image.Image,
    match_size: bool = True,
    resample = Image.Resampling.BICUBIC,
    offset: int = 0,                        # biais ajouté après addition (remonte la luminance)
    k = 210
) -> Image.Image:
    """
    Additionne img2 à img1 : out = clip(img1 + img2 + offset, 0..255)
    - match_size : redimensionne img2 à la taille d'img1 si nécessaire
    - offset : ajoute un biais après addition
    Remarque : ImageChops.add clippe automatiquement à 255.
    """

    # Unifier tailles
    if match_size and (img2.size != img1.size):
        img2 = img2.resize(img1.size, resample=resample)

    # Travailler en RGB (simple et sûr)
    work_mode = "RGB"
    im1 = img1.convert(work_mode)
    im2 = img2.convert(work_mode)

    # Enlever k à tous les pixels de im2
    im2 = subtract_constant_clip0(im2, k=k)

    # Addition + offset (clip 0..255 géré par Pillow)
    out_rgb = ImageChops.add(im1, im2, scale=1.0, offset=offset)

    return out_rgb


def grey_mask(
    img1: Image.Image,
    img2: Image.Image,
    match_size: bool = True,
    resample = Image.Resampling.NEAREST,   # NEAREST conseillé pour les masques nets
    offset: int = 0,                       # (conservé pour compat, non utilisé ici)
    *,
    threshold: int | None = None,          # ex. 128 pour binariser le masque
    feather: float = 0.0,                  # flou gaussien sur le masque (adoucit les bords)
    invert_mask: bool = False,             # True => gris à l'extérieur du masque
    # --- options de contour ---
    outline_width: int = 0,                # 0 = pas de contour ; >0 en pixels
    outline_color: tuple[int, int, int] = (255, 0, 0),
    outline_opacity: int = 255,            # 0..255
    outline_feather: float = 0.0           # flou du contour (soft stroke)
) -> Image.Image:
    """
    Grise img1 dans les zones définies par img2 (utilisée comme masque).
    Ajoute en option un CONTOUR autour des bords du masque.

    - threshold : binarise le masque (>= seuil -> 255, sinon 0)
    - feather   : adoucit la transition du masque (flou gaussien)
    - outline_* : contrôle du contour (épaisseur, couleur, opacité, adoucissement)
    """
    # 1) Préparer et aligner le masque à img1
    mask = img2.convert("L")
    if match_size and mask.size != img1.size:
        mask = mask.resize(img1.size, resample=resample)

    if threshold is not None:
        t = max(0, min(255, int(threshold)))
        mask = mask.point(lambda v: 255 if v >= t else 0, mode="L")

    if invert_mask:
        mask = ImageOps.invert(mask)

    if feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))

    # 2) Version grise de img1 (on préserve l’alpha si présent)
    if img1.mode == "RGBA":
        base = img1
        gL = ImageOps.grayscale(img1)                  # "L"
        gray = Image.merge("LA", (gL, img1.getchannel("A"))).convert("RGBA")
    else:
        base = img1.convert("RGB")
        gray = ImageOps.grayscale(base).convert("RGB")

    # 3) Composer : zones du masque -> gris, ailleurs -> original
    out = Image.composite(gray, base, mask)

    # 4) (Optionnel) Ajouter un contour autour du masque
    if outline_width and outline_width > 0 and outline_opacity > 0:
        # Binariser (au cas où) pour un bord net
        binmask = mask.point(lambda v: 255 if v > 0 else 0, mode="L")

        # Taille de noyau (impair) pour Max/MinFilter
        k = max(1, int(outline_width))
        size = max(3, 2 * k + 1)

        # Dilatation & érosion -> bord = dilate - erode (anneau)
        dil = binmask.filter(ImageFilter.MaxFilter(size))
        ero = binmask.filter(ImageFilter.MinFilter(size))
        edge = ImageChops.difference(dil, ero)     # anneau autour du front
        # Nettoyage
        edge = edge.point(lambda v: 255 if v > 0 else 0, mode="L")

        # Adoucir le contour si demandé
        if outline_feather and outline_feather > 0:
            edge = edge.filter(ImageFilter.GaussianBlur(float(outline_feather)))

        # Préparer le calque “trait”
        stroke_rgba = Image.new("RGBA", out.size, (*outline_color, int(outline_opacity)))

        # S’assurer que la base est en RGBA pour composer
        out_rgba = out.convert("RGBA")
        # Appliquer le trait là où edge > 0
        out = Image.composite(stroke_rgba, out_rgba, edge)

        # Si l’image d’origine n’avait pas d’alpha et que tu préfères revenir en RGB:
        if img1.mode != "RGBA":
            out = out.convert("RGB")

    return out

def save_tiles(out, tiles, folder='tiles', mask_mode='grey_watermark', split_color_mode=None):

    # Sauvegarder chaque tuile
    for i, t in enumerate(tiles):
        t.save(f"{folder}/tile_{i:02d}.png")

    # Sauvegarder l'image reconstituée
    out.save(f"{folder}/color6.png")
    
    if split_color_mode == 'grey_even' and mask_mode != 'grey_watermark':
        for i, t in enumerate(tiles):
            if i % 2 == 0:  # change à 1 si tu veux (1,3,5,...)
                if t.mode == "RGBA":
                    # garder l’alpha
                    g = ImageOps.grayscale(t)        # "L"
                    tiles[i] = Image.merge("LA", (g, t.getchannel("A"))).convert("RGBA")
                else:
                    tiles[i] = ImageOps.grayscale(t).convert("RGB")

            tiles[i].save(f"{folder}/tile_{i:02d}.png")

        # recoller les tuiles dans une image finale
        rows, cols = 2, 3
        W, H = out.size
        x_edges = [round(W * c / cols) for c in range(cols + 1)]
        y_edges = [round(H * r / rows) for r in range(rows + 1)]
        canvas = Image.new(out.mode, (W, H))

        k = 0
        for r in range(rows):
            for c in range(cols):
                canvas.paste(tiles[k], (x_edges[c], y_edges[r]))
                k += 1

        canvas.save(f"{folder}/grey6.png")


def create_random_white_mask(
    black_bg: Image.Image,               # ex: Image.new("RGBA", base.size, (0,0,0,255))
    whited: Image.Image,                 # mask « blanc » (ou n’importe quelle image -> on prend sa luminance)
    n: int = 8,
    scale_range: Tuple[float, float] = (0.6, 1.3),   # facteur d'échelle min/max
    angle_range: Tuple[float, float] = (-20.0, 20.0),# rotation aléatoire (degrés)
    threshold: int | None = None,        # None = niveaux de gris; sinon binaire (>=t -> opaque)
    feather_edge: float = 0.0,           # lisser le bord (flou gaussien sur l’alpha)
    resample_scale = Image.Resampling.LANCZOS,
    resample_rotate = Image.Resampling.BICUBIC,
    seed: int | None = None,
) -> Image.Image:
    """
    Accumule (ADD) n couches blanches issues de `whited` sur un fond noir `black_bg`.

    - À chaque itération:
        1) alpha = luminance(whited)  [seuil/feather optionnels]
        2) couche RGBA blanche (255,255,255, alpha)
        3) scale + rotate (expand=True)
        4) placement (x,y) aléatoire dans le cadre
        5) accumulation par ADD (clip 255) sur un canevas 'acc'

    - Retourne une image RGBA : black_bg + acc.
    """
    if seed is not None:
        random.seed(seed)

    # S'assure que le fond et le canevas ont la même taille
    BW, BH = black_bg.size
    base = black_bg.convert("RGBA")
    acc  = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))

    # Préparer la source alpha (on n'utilise QUE l'alpha dérivé du mask)
    src_alpha = whited.convert("L")
    if threshold is not None:
        t = max(0, min(255, int(threshold)))
        src_alpha = src_alpha.point(lambda v: 255 if v >= t else 0, mode="L")

    if feather_edge and feather_edge > 0:
        src_alpha = src_alpha.filter(ImageFilter.GaussianBlur(float(feather_edge)))

    # Couche blanche “canonique” (RGB = blanc, alpha = src_alpha)
    # On crée une fonction utilitaire pour construire la couche (utile après resize/rotate)
    def make_white_layer(alpha_img: Image.Image) -> Image.Image:
        Lw, Lh = alpha_img.size
        layer = Image.new("RGBA", (Lw, Lh), (255, 255, 255, 0))
        layer.putalpha(alpha_img)
        return layer

    # Boucle d'accumulation
    for _ in range(max(0, int(n))):
        # 1) Scale aléatoire de l'alpha
        s = random.uniform(*scale_range)
        a_scaled = src_alpha.resize(
            (max(1, int(src_alpha.width  * s)),
             max(1, int(src_alpha.height * s))),
            resample=resample_scale
        )

        # 2) Rotation aléatoire (expand=True) – on tourne l'alpha
        angle = random.uniform(*angle_range)
        a_rot = a_scaled.rotate(angle, resample=resample_rotate, expand=True, fillcolor=0)

        # 3) Couche blanche correspondante
        layer = make_white_layer(a_rot)
        LW, LH = layer.size

        # 4) Si la couche ne tient pas, on la redimensionne pour qu'elle rentre
        if LW > BW or LH > BH:
            fit = min(BW / LW, BH / LH)
            if fit < 1:
                newW = max(1, int(LW * fit))
                newH = max(1, int(LH * fit))
                a_rot = a_rot.resize((newW, newH), resample=resample_scale)
                layer = make_white_layer(a_rot)
                LW, LH = layer.size

        if LW > BW or LH > BH:
            continue  # cas extrême, on saute

        # 5) Coordonnées aléatoires valides
        x = random.randint(0, BW - LW) if BW > LW else 0
        y = random.randint(0, BH - LH) if BH > LH else 0

        # 6) Accumulation ADD : on fabrique un patch pleine taille et on ADD par canal
        patch = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
        patch.alpha_composite(layer, dest=(x, y))

        r1, g1, b1, a1 = acc.split()
        r2, g2, b2, a2 = patch.split()
        r = ImageChops.add(r1, r2)   # addition, clip 255
        g = ImageChops.add(g1, g2)
        b = ImageChops.add(b1, b2)
        # Pour l’alpha : on prend le max (préserve les contours déjà en place)
        a = ImageChops.lighter(a1, a2)

        acc = Image.merge("RGBA", (r, g, b, a))

    # Pose finale sur le fond noir (utile si black_bg n'est pas pur noir opaque)
    out = base.copy()
    out.alpha_composite(acc)
    return out

def apply_composite_mask(
    base: Image.Image,
    blacked: Image.Image,
    *,
    match_size: bool = True,
    resample = Image.Resampling.NEAREST,  # NEAREST = masque net ; BICUBIC si tu redimensionnes du gris
    threshold: int | None = 200,          # binarise le masque : >= seuil -> 255 (blanc=base), < seuil -> 0 (noir=fond)
    feather: float = 0.0,                 # adoucit le bord du masque (flou gaussien)
    invert_mask: bool = False,            # inverse la logique si besoin
    white_color: tuple[int,int,int] = (255, 255, 255)  # couleur de fond à la place du “noir”
) -> Image.Image:
    """
    Sortie:
      - zones BLANCHES de `blacked` => image `base`
      - zones NOIRES   de `blacked` => `white_color` (par défaut: blanc)

    Paramètres utiles:
      - threshold: None pour garder un masque en niveaux de gris (transition douce),
                   ou une valeur 0..255 pour un bord net (binaire).
      - feather: >0 pour flouter le masque et créer des contours plus doux.
    """
    # 1) Préparer le masque à partir de `blacked`
    mask = blacked.convert("L")

    if threshold is not None:
        t = max(0, min(255, int(threshold)))
        mask = mask.point(lambda v: 255 if v >= t else 0, mode="L")

    if invert_mask:
        mask = ImageOps.invert(mask)

    if feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))

    # 2) Préparer les deux “couches” : base et fond blanc (ou autre couleur)
    base_rgb = base.convert("RGB")
    white_bg = Image.new("RGB", base_rgb.size, white_color)

    # 3) Composer: là où mask est clair → prend base, sinon → prend white_bg
    out = Image.composite(base_rgb, white_bg, mask)
    return out