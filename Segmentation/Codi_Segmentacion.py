##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI PER FER LA SEGMENTACIÓ DELS MINERALS AMB SAM2

##############################################################################################################

import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DATASET_ROOT = "../Dataset/original_videos"
FACT_BASE = "https://www.virtualmicroscope.org/content"
OUTPUT_ROOT = "../Dataset/segmented_minerals"

datasets = [

("Hornblende","I08","jrs004",1),
("Hornblende","I09","jrs006",1),
("Hornblende","I09","jrs006",2),
("Hornblende","M01","p5067",1),
("Hornblende","M01","p5067",2),
("Hornblende","hornblende_porphyrite_blue_mountains","cd2142",1),
("Hornblende","hornblende_porphyrite_valparaiso","cd2142",1),
("Hornblende","hornblende_porphyrite_valparaiso","cd2142",2),
("Hornblende","diorite_magellan_straits","cd2079",1),
("Hornblende","hornblende_schist","cd0660",1),
("Hornblende","amygdaloidal_monchiquite","cd0261",1),
("Hornblende","amygdaloidal_monchiquite","cd0261",2),
("Hornblende","amygdaloidal_monchiquite","cd0261",3),
("Hornblende","phonolite_cape_verde","cd0164",1),
("Hornblende","epidote_amphibolite_mayo","p5468",1),
("Hornblende","epidote_amphibolite_mayo","p5468",2),
("Hornblende","I16","p19184",1),
("Hornblende","I16","p19184",2),
("Hornblende","M26","p19181",2),
("Hornblende","felsite_isle_rum","du13876",1),
("Hornblende","felsite_isle_rum","du9907",2),
("Hornblende","granophyre_isle_rum","du9904",1),

("Biotite","I08","jrs004",2),
("Biotite","I09","jrs006",1),
("Biotite","I09","jrs006",2),
("Biotite","nepheline_syenite_uk","nephsyenite1",2),
("Biotite","nepheline_syenite_uk","nephsyenite1",3),
("Biotite","green_schist","p5049",1),
("Biotite","green_schist","p5049",2),
("Biotite","garnet_mica_schist_galway","p5215",1),
("Biotite","garnet_mica_schist_galway","p5215",3),
("Biotite","garnet_kyanite_gneiss_slishwood","p5473",1),
("Biotite","granophyre_mourne","ig29",1),
("Biotite","granophyre_mourne","ig29",2),
("Biotite","granite_leinster","ig1",1),
("Biotite","granite_leinster","ig1",2),
("Biotite","siderophyllite_granite_colcerrow33","12agt33",1),
("Biotite","siderophyllite_granite_colcerrow33","12agt33",2),
("Biotite","siderophyllite_granite_pelyn32","12agt32",3),
("Biotite","siderophyllite_granite_luxulyan2","137201",1),
("Biotite","siderophyllite_granite_luxulyan2","137201",2),
("Biotite","siderophyllite_granite_luxulyan2","137201",3),
("Biotite","M02","p5468",1),
("Biotite","M02","p5468",2),
("Biotite","siderophyllite_granite_luxulyan1","11agt10",2),
("Biotite","altered_granite_schist_contact","cd3668",1),
("Biotite","garnet_biotite_augen_gneiss","cd0468",1),
("Biotite","garnet_biotite_augen_gneiss","cd0468",2),
("Biotite","myrmekitic_granulite_charnockite","cd0311",1),
("Biotite","M09","p5563",1),
("Biotite","M09","p5563",2),
("Biotite","granite_I04","tc_p5471",1),
("Biotite","quartz_monzonite_jersey","smg490",2),
("Biotite","sample_s339_20","s339_20",1),
("Biotite","sample_s339_20","s339_20",2),
("Biotite","sample_s339_20","s339_20",3),
("Biotite","sample_glahm134581","glahm134581",1),
("Biotite","sample_glahm134581","glahm134581",2),
("Biotite","sample_bgs_59192","bgs_59192",2),
("Biotite","sample_11agt16","11agt16",1),
("Biotite","shap_granite","shap_gr",1),
("Biotite","shap_granite","shap_gr",2),
("Biotite","generic_granite","granite",1),
("Biotite","generic_granite","granite",2),
("Biotite","generic_granite","granite",3),
("Biotite","generic_granite","granite",4),

]

sam2_model = build_sam2(
    "sam2_hiera_l.yaml",
    "sam2_hiera_large.pt",
    device="cpu"
)

predictor = SAM2ImagePredictor(sam2_model)

# Funcio per imprimir la descripcio de la roca
def get_fact_sheet(asset_id):

    """
    https://www.virtualmicroscope.org/content/uk-nepheline-syenite
    """

    try:

        url = f"{FACT_BASE}/{asset_id}"

        r = requests.get(url,timeout=10)
        soup = BeautifulSoup(r.text,"html.parser")

        desc = soup.find("div",class_="field--name-field-description")

        if desc:
            return desc.get_text("\n",strip=True)

    except Exception as e:
        print("Fact sheet error:",e)

    return "Fact sheet not available"

# Carregar els VIDEO FRAMES

def load_frames(folder):

    files = sorted(os.listdir(folder))

    frames = []

    for f in files:

        if f.endswith(".jpg"):

            img = cv2.imread(os.path.join(folder,f))

            frames.append(img)

    return frames

############################################
# ROTATEM ELS FRAMES
############################################

def rotate_frame(frame, angle):
    if abs(angle) < 1e-3:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    # BORDER_REFLECT disimula los bordes negros en rotaciones pequeñas
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
def rotate_frame_high_quality(frame, angle):
    if abs(angle) < 0.01: return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    # INTER_CUBIC es clave para evitar que se vea "borroso" al girar
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

# Funcio per aplicar mascara

def apply_mask(frame,mask):

    white = np.ones_like(frame)*255

    mask3 = np.repeat(mask[:,:,None],3,axis=2)

    result = np.where(mask3,frame,white)

    return result

# Funcio per retallar el mineral amb la mascara

def crop_from_mask(frame,mask):

    ys,xs = np.where(mask)

    x1,x2 = xs.min(),xs.max()
    y1,y2 = ys.min(),ys.max()

    return frame[y1:y2,x1:x2]

# guardem el nou video
def save_video(frames,path):

    h,w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(path,fourcc,10,(w,h))

    for f in frames:

        out.write(f)

    out.release()

# funcio per segmentar
def segment(frame,roi):

    x,y,w,h = roi

    box = np.array([[x,y,x+w,y+h]])

    predictor.set_image(frame)

    masks,_,_ = predictor.predict(box=box)

    mask = masks[0]

    return mask


def run_segmentation(ppl_frames, xpl_frames, fact_sheet, mineral, video_name):

    angle = 0.0
    total_frames = min(len(ppl_frames), len(xpl_frames))
    ppl_frames = ppl_frames[:total_frames]
    xpl_frames = xpl_frames[:total_frames]
    step = 360.0 / total_frames

    mask = None
    roi = None
    pending_roi = None
    crystal_id = 1

    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmentation", 1500, 900)

    h_orig, w_orig = ppl_frames[0].shape[:2]

    text_panel = np.zeros((h_orig, 420, 3), dtype=np.uint8)

    y = 30

    instructions = [
        "CONTROLS",
        "",
        "R  -> Select ROI",
        "ENTER -> Run segmentation",
        "C  -> Cancel segmentation",
        "S  -> Save segmented crystal",
        "P  -> Preview 72-frame rotation",
        "N  -> Next rock",
        "ESC -> Exit",
        "",
        "Drag mouse up/down",
        "to rotate platina",
        "",
        "FACT SHEET"
    ]

    for line in instructions:
        cv2.putText(text_panel, line, (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,255),1)
        y += 25

    for line in fact_sheet.split("\n"):
        cv2.putText(text_panel,line[:55],(10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(200,200,200),1)
        y += 20
        if y > h_orig-20:
            break


    dragging = False
    last_y = 0

    def mouse_callback(event,x,y,flags,param):

        nonlocal dragging,last_y,angle

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last_y = y

        elif event == cv2.EVENT_MOUSEMOVE and dragging:

            dy = y - last_y
            last_y = y

            angle = (angle - dy*0.2) % 360

        elif event == cv2.EVENT_LBUTTONUP:

            dragging = False


    cv2.setMouseCallback("Segmentation", mouse_callback)


    while True:

        frame_idx = int(angle // step) % total_frames

        ppl_frame = ppl_frames[frame_idx].copy()
        xpl_frame = xpl_frames[frame_idx].copy()

        angle_display = frame_idx * step

        ppl_view = ppl_frame.copy()
        xpl_view = xpl_frame.copy()


        rotated_mask = None

        if mask is not None:

            rotated_mask = mask.astype(np.uint8)

            ppl_masked = apply_mask(ppl_frame, rotated_mask)
            xpl_masked = apply_mask(xpl_frame, rotated_mask)

            if roi is not None:
                x, y, w_box, h_box = roi
                x1, y1 = x, y
                x2, y2 = x + w_box, y + h_box
            else:
                ys, xs = np.where(mask)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

            ppl_view = ppl_masked[y1:y2, x1:x2]
            xpl_view = xpl_masked[y1:y2, x1:x2]

            dw, dh = 600, 450

            ppl_view = cv2.resize(ppl_view, (dw, dh))
            xpl_view = cv2.resize(xpl_view, (dw, dh))

            contours,_ = cv2.findContours(mask.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(ppl_frame,contours,-1,(0,255,0),2)
            cv2.drawContours(xpl_frame,contours,-1,(0,255,0),2)



        frame0 = ppl_frames[0].copy()

        if pending_roi is not None:
            x,y,w,h = pending_roi
            cv2.rectangle(frame0,(x,y),(x+w,y+h),(0,255,255),2)

        if mask is not None:

            contours,_ = cv2.findContours(mask.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(frame0,contours,-1,(0,255,0),2)


        cv2.putText(ppl_view,f"{angle_display:.1f} deg",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        ph, pw = ppl_view.shape[:2]
        cx, cy = pw // 2, ph // 2

        # PPL
        cv2.line(ppl_view, (cx,0), (cx,ph), (255,0,0), 1)
        cv2.line(ppl_view, (0,cy), (pw,cy), (255,0,0), 1)

        # XPL
        xh, xw = xpl_view.shape[:2]
        cx2, cy2 = xw // 2, xh // 2

        cv2.line(xpl_view, (cx2,0), (cx2,xh), (255,0,0), 1)
        cv2.line(xpl_view, (0,cy2), (xw,cy2), (255,0,0), 1)

        dw, dh = 600, 450

        canvas_top = np.hstack([
            ppl_view if ppl_view.shape[1] == dw else cv2.resize(ppl_view,(dw,dh)),
            xpl_view if xpl_view.shape[1] == dw else cv2.resize(xpl_view,(dw,dh))
        ])

        canvas_bottom = np.hstack([
            cv2.resize(frame0, (dw, dh)),
            cv2.resize(text_panel, (dw, dh))
        ])

        canvas = np.vstack([canvas_top, canvas_bottom])

        title = f"ROCK: {video_name}"

        cv2.putText(
            canvas,
            title,
            (475, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2
        )

        cv2.imshow("Segmentation", canvas)


        key = cv2.waitKey(10) & 0xFF


        if key == ord('r'):

            pending_roi = cv2.selectROI("Segmentation",
                                        ppl_frames[0],
                                        False)

            cv2.setMouseCallback("Segmentation", mouse_callback)


        elif key == 13 or key == 10:

            if pending_roi is not None:

                mask = segment(ppl_frames[0],pending_roi)

                roi = pending_roi

                pending_roi = None


        elif key == ord('c'):

            mask = None
            roi = None
            pending_roi = None

        elif key == ord('p'):

            # Preview
            if mask is not None:

                total = len(ppl_frames)
                step_preview = 360.0 / total

                h, w = mask.shape
                center = (w // 2, h // 2)

                for i in range(total):

                    angle_p = i * step_preview

                    M = cv2.getRotationMatrix2D(center, -angle_p, 1.0)

                    rotated_mask = cv2.warpAffine(
                        mask.astype(np.uint8),
                        M,
                        (w, h),
                        flags=cv2.INTER_NEAREST
                    )

                    p = ppl_frames[i].copy()
                    x = xpl_frames[i].copy()

                    p = apply_mask(p, rotated_mask)
                    x = apply_mask(x, rotated_mask)

                    preview = np.hstack([
                        cv2.resize(p,(600,450)),
                        cv2.resize(x,(600,450))
                    ])

                    cv2.putText(preview,
                                f"Rotation preview {i+1}/{total}",
                                (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(0,255,0),2)

                    cv2.imshow("Segmentation", preview)

                    if cv2.waitKey(40) & 0xFF == 27:
                        break

        elif key == ord('s'):

            if mask is not None:

                save_segmented_video(
                    ppl_frames,
                    xpl_frames,
                    mask,
                    mineral,
                    video_name,
                    crystal_id
                )

                crystal_id += 1

                print("Crystal saved")


        elif key == ord('n'):

            print("Next rock")
            break


        elif key == 27:

            break


    cv2.destroyAllWindows()


# funcio per guardar el video segmentat
def save_segmented_video(
        ppl_frames,
        xpl_frames,
        mask,
        mineral,
        video_name,
        crystal_id
):

    out_dir = os.path.join(
        OUTPUT_ROOT,
        mineral,
        f"{video_name}_crystal_{crystal_id}"
    )

    ppl_dir = os.path.join(out_dir, "PPL_frames")
    xpl_dir = os.path.join(out_dir, "XPL_frames")

    os.makedirs(ppl_dir, exist_ok=True)
    os.makedirs(xpl_dir, exist_ok=True)

    total_frames = len(ppl_frames)
    step = 360.0 / total_frames

    h, w = mask.shape
    center = (w // 2, h // 2)

    for i, (p, x) in enumerate(zip(ppl_frames, xpl_frames)):

        angle = i * step

        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        rotated_mask = cv2.warpAffine(
            mask.astype(np.uint8),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST
        )

        seg_p = apply_mask(p, rotated_mask)
        seg_x = apply_mask(x, rotated_mask)

        cv2.imwrite(
            os.path.join(ppl_dir, f"frame_{i:03d}.png"),
            seg_p
        )

        cv2.imwrite(
            os.path.join(xpl_dir, f"frame_{i:03d}.png"),
            seg_x
        )

    print("Saved 72 segmented frames to:", out_dir)

# MAIN

for mineral,folder,asset,rotation in datasets:

    print("Processing",folder,"rotation",rotation)

    # Estructura es la següent :

    # original_videos/
    #    Mineral/
    #        PPL/
    #            nomroca_rotX/
    #        XPL/
    #            nomroca_rotX/

    video_folder = f"{folder}_rot{rotation}"

    ppl_dir = os.path.join(DATASET_ROOT, mineral, "PPL", video_folder)
    xpl_dir = os.path.join(DATASET_ROOT, mineral, "XPL", video_folder)

    if not os.path.exists(ppl_dir):
        print("Missing folder:", ppl_dir)
        print("Expected structure: original_videos/<Mineral>/PPL/<rock>_rotX/")
        continue

    ppl_frames = load_frames(ppl_dir)
    xpl_frames = load_frames(xpl_dir)

    fact = get_fact_sheet(asset)

    run_segmentation(
        ppl_frames,
        xpl_frames,
        fact,
        mineral,
        f"{folder}_rot{rotation}"
    )