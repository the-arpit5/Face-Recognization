import tkinter as tk
from tkinter import ttk, filedialog, messagebox as mess
import cv2, os, csv, numpy as np, pandas as pd, datetime, time
from PIL import Image
import tkinter.simpledialog as tsd

# --- Directory Setup ---
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

assure_path_exists("StudentDetails/")
assure_path_exists("TrainingImage/")
assure_path_exists("TrainingImageLabel/")

def tick():
    clock.config(text=time.strftime('%H:%M:%S'))
    clock.after(200, tick)

def check_haarcascade():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror('Error', 'haarcascade_frontalface_default.xml file missing!')
        window.destroy()

# --- Registration Logic ---
def UploadAndRegister():
    check_haarcascade()
    Id, name = txt_id.get().strip(), txt_name.get().strip()
    
    if not Id.isdigit() or not name.replace(' ','').isalpha():
        return mess.showerror("Input Error", "ID (Numbers) aur Name (Alphabets) sahi bharein!")

    file_path = filedialog.askopenfilename(title="Select Student Photo", 
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            return mess.showerror("Error", "Photo load nahi ho saki!")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return mess.showwarning("No Face", "Is photo mein chehra nahi mila!")

        csv_path = "StudentDetails/StudentDetails.csv"
        serial = 1
        if os.path.isfile(csv_path) and os.stat(csv_path).st_size > 0:
            with open(csv_path, 'r') as f:
                serial = len(list(csv.reader(f)))

        for (x, y, w, h) in faces:
            # AI training ke liye 30 samples (Jyada samples = Better recognition)
            for i in range(1, 31):
                cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{i}.jpg", gray[y:y+h, x:x+w])
            break 
        
        with open(csv_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SERIAL NO.', 'ID', 'NAME', 'DATE', 'TIME'])
            writer.writerow([serial, Id, name, '', ''])
            
        lbl_status.configure(text=f"Status: {name} Saved!", fg="#16a34a")
        mess.showinfo("Success", f"{name} Register ho gaya. Ab Step 2 dabayein!")
        txt_id.delete(0, 'end'); txt_name.delete(0, 'end')

# --- Training Logic ---
def TrainImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        img_path = "TrainingImage"
        imagePaths = [os.path.join(img_path, f) for f in os.listdir(img_path)]
        faces, ids = [], []
        for p in imagePaths:
            faces.append(np.array(Image.open(p).convert('L'), 'uint8'))
            # Filename se Serial No nikalna (Name.Serial.ID.Sample.jpg)
            serial_num = int(os.path.split(p)[-1].split(".")[1])
            ids.append(serial_num)
        
        recognizer.train(faces, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainner.yml")
        mess.showinfo("Success", "AI System Train Ho Gaya!")
    except Exception as e: mess.showerror("Error", f"Training Failed: {str(e)}")

# --- Scanner Logic (FIXED: Improved Threshold) ---
def TrackImages():
    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        return mess.showwarning("Error", "Pehle Training karein!")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    csv_path = "StudentDetails/StudentDetails.csv"
    df = pd.read_csv(csv_path)
    # Column names ko clean karna
    df.columns = df.columns.str.strip().str.upper()

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, im = cam.read()
        if not ret: break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            
            # conf < 75 matlab 25% match bhi hoga toh naam dikhayega (Photo upload ke liye zaroori hai)
            if conf < 75: 
                match = df.loc[df['SERIAL NO.'] == serial]
                if not match.empty:
                    name = str(match['NAME'].values[0])
                    # Attendance mark karna
                    date = datetime.datetime.now().strftime('%d-%m-%Y')
                    ts = datetime.datetime.now().strftime('%H:%M:%S')
                    df.loc[df['SERIAL NO.'] == serial, 'DATE'] = date
                    df.loc[df['SERIAL NO.'] == serial, 'TIME'] = ts
                    df.to_csv(csv_path, index=False)
                    
                    cv2.putText(im, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(im, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Attendance Scanner - Press Q to Exit', im)
        if cv2.waitKey(1) == ord('q'): break
    
    cam.release(); cv2.destroyAllWindows(); LoadAttendance()

def LoadAttendance():
    for k in tv.get_children(): tv.delete(k)
    csv_path = "StudentDetails/StudentDetails.csv"
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df_marked = df.dropna(subset=['DATE'])
        for _, row in df_marked.iterrows():
            tv.insert('', 0, values=(row['ID'], row['NAME'], row['DATE'], row['TIME']))

# --- UI Layout (Same as before) ---
window = tk.Tk()
window.geometry("900x650")
window.title("Attendance System - Final Fix")
window.configure(background="#f8fafc")

header = tk.Frame(window, bg="#1e293b", height=80)
header.pack(fill="x")
tk.Label(header, text="SMART ATTENDANCE", bg="#1e293b", fg="white", font=('Arial', 20, 'bold')).pack(pady=15)
clock = tk.Label(header, bg="#1e293b", fg="#38bdf8", font=('Arial', 12, 'bold'))
clock.place(x=780, y=30); tick()

tabs = ttk.Notebook(window)
tab1, tab2, tab3 = tk.Frame(tabs, bg="white"), tk.Frame(tabs, bg="white"), tk.Frame(tabs, bg="white")
tabs.add(tab1, text=" REGISTRATION "); tabs.add(tab2, text=" SCANNER "); tabs.add(tab3, text=" RECORDS ")
tabs.pack(expand=1, fill="both", padx=10, pady=10)

# Tab 1
txt_id = tk.Entry(tab1, font=('Arial', 12), width=30, bd=1, relief="solid"); txt_id.pack(pady=5)
txt_name = tk.Entry(tab1, font=('Arial', 12), width=30, bd=1, relief="solid"); txt_name.pack(pady=5)
tk.Button(tab1, text="STEP 1: UPLOAD & SAVE", command=UploadAndRegister, bg="#4f46e5", fg="white", font=('Arial', 10, 'bold'), width=35, pady=12).pack(pady=20)
tk.Button(tab1, text="STEP 2: TRAIN AI", command=TrainImages, bg="#16a34a", fg="white", font=('Arial', 10, 'bold'), width=35, pady=12).pack()
lbl_status = tk.Label(tab1, text="Status: Ready", bg="white", fg="gray"); lbl_status.pack(pady=20)

# Tab 2
tk.Button(tab2, text="LAUNCH SCANNER", command=TrackImages, bg="#4f46e5", fg="white", font=('Arial', 12, 'bold'), width=30, pady=30).pack(pady=100)

# Tab 3
tv = ttk.Treeview(tab3, columns=('id','name','date','time'), show='headings')
for col, head in zip(['id','name','date','time'], ['ID', 'NAME', 'DATE', 'TIME']):
    tv.heading(col, text=head); tv.column(col, width=150)
tv.pack(fill="both", expand=True, padx=20, pady=20)

LoadAttendance()
window.mainloop()

# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox as mess
# import cv2, os, csv, numpy as np, pandas as pd, datetime, time
# from PIL import Image
# import tkinter.simpledialog as tsd

# # --- Directory Setup ---
# def assure_path_exists(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# assure_path_exists("StudentDetails/")
# assure_path_exists("TrainingImage/")
# assure_path_exists("TrainingImageLabel/")

# # --- Helper Functions ---
# def tick():
#     clock.config(text=time.strftime('%H:%M:%S'))
#     clock.after(200, tick)

# def check_haarcascade():
#     if not os.path.isfile("haarcascade_frontalface_default.xml"):
#         mess.showerror('Error', 'haarcascade_frontalface_default.xml file missing!')
#         window.destroy()

# # --- Registration Logic (Upload Photo) ---
# def UploadAndRegister():
#     check_haarcascade()
#     Id, name = txt_id.get().strip(), txt_name.get().strip()
    
#     if not Id.isdigit() or not name.replace(' ','').isalpha():
#         return mess.showerror("Input Error", "ID (Numbers) aur Name (Alphabets) sahi bharein!")

#     # Photo select karne ke liye window khulegi
#     file_path = filedialog.askopenfilename(title="Select Student Photo", 
#                                           filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
#     if file_path:
#         img = cv2.imread(file_path)
#         if img is None:
#             return mess.showerror("Error", "Photo load nahi ho saki!")
            
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         if len(faces) == 0:
#             return mess.showwarning("No Face", "Is photo mein chehra nahi mila! Dusri photo try karein.")

#         # Serial Number calculation
#         csv_path = "StudentDetails/StudentDetails.csv"
#         serial = 1
#         if os.path.isfile(csv_path) and os.stat(csv_path).st_size > 0:
#             with open(csv_path, 'r') as f:
#                 serial = len(list(csv.reader(f)))

#         # Sirf pehla chehra save karein
#         for (x, y, w, h) in faces:
#             # Training ke liye 10 samples generate kar rahe hain (Best accuracy ke liye)
#             for i in range(1, 11):
#                 cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{i}.jpg", gray[y:y+h, x:x+w])
#             break 
        
#         # CSV mein Entry
#         with open(csv_path, 'a+', newline='') as f:
#             writer = csv.writer(f)
#             if os.stat(csv_path).st_size == 0:
#                 writer.writerow(['SERIAL NO.', 'ID', 'NAME', 'DATE', 'TIME'])
#             writer.writerow([serial, Id, name, '', ''])
            
#         lbl_status.configure(text=f"Status: {name} Registered from Photo!", fg="#16a34a")
#         mess.showinfo("Success", f"{name} ka registration upload ke zariye ho gaya!")
#         txt_id.delete(0, 'end'); txt_name.delete(0, 'end')

# # --- Training Logic ---
# def TrainImages():
#     path = "TrainingImageLabel/psd.txt"
#     if not os.path.isfile(path):
#         new_pas = tsd.askstring('Setup', 'Set Admin Password:', show='*')
#         if new_pas: 
#             with open(path, "w") as f: f.write(new_pas)
    
#     password = tsd.askstring('Admin Access', 'Enter Password:', show='*')
#     if os.path.isfile(path):
#         with open(path, "r") as f:
#             if password != f.read(): return mess.showerror('Error', 'Wrong Password!')

#     try:
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         img_path = "TrainingImage"
#         imagePaths = [os.path.join(img_path, f) for f in os.listdir(img_path)]
#         faces, ids = [], []
#         for p in imagePaths:
#             faces.append(np.array(Image.open(p).convert('L'), 'uint8'))
#             ids.append(int(os.path.split(p)[-1].split(".")[1]))
        
#         recognizer.train(faces, np.array(ids))
#         recognizer.save("TrainingImageLabel/Trainner.yml")
#         mess.showinfo("Success", "AI System Updated!")
#     except Exception as e: mess.showerror("Error", str(e))

# # --- Scanner & View Logic ---
# def TrackImages():
#     if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
#         return mess.showwarning("Error", "Pehle Training karein!")
    
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel/Trainner.yml")
#     faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
#     csv_path = "StudentDetails/StudentDetails.csv"
#     df = pd.read_csv(csv_path)
#     df.columns = df.columns.str.strip().str.upper()

#     cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     while True:
#         ret, im = cam.read()
#         if not ret: break
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#         for (x, y, w, h) in faces:
#             serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
#             if conf < 75:
#                 name = str(df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0])
#                 date = datetime.datetime.now().strftime('%d-%m-%Y')
#                 ts = datetime.datetime.now().strftime('%H:%M:%S')
#                 df.loc[df['SERIAL NO.'] == serial, 'DATE'] = date
#                 df.loc[df['SERIAL NO.'] == serial, 'TIME'] = ts
#                 df.to_csv(csv_path, index=False)
#                 cv2.putText(im, f"{name} Marked", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#             cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.imshow('Attendance Scanner - Press Q to Exit', im)
#         if cv2.waitKey(1) == ord('q'): break
#     cam.release(); cv2.destroyAllWindows(); LoadAttendance()

# def LoadAttendance():
#     for k in tv.get_children(): tv.delete(k)
#     csv_path = "StudentDetails/StudentDetails.csv"
#     if os.path.isfile(csv_path):
#         df = pd.read_csv(csv_path)
#         df_marked = df.dropna(subset=['DATE'])
#         for _, row in df_marked.iterrows():
#             tv.insert('', 0, values=(row['ID'], row['NAME'], row['DATE'], row['TIME']))

# # --- UI Layout ---
# window = tk.Tk()
# window.geometry("900x650")
# window.title("Attendance System - Photo Upload Mode")
# window.configure(background="#f8fafc")

# header = tk.Frame(window, bg="#1e293b", height=80)
# header.pack(fill="x")
# tk.Label(header, text="SMART ATTENDANCE", bg="#1e293b", fg="white", font=('Arial', 20, 'bold')).pack(pady=15)
# clock = tk.Label(header, bg="#1e293b", fg="#38bdf8", font=('Arial', 12, 'bold'))
# clock.place(x=780, y=30); tick()

# tabs = ttk.Notebook(window)
# tab1, tab2, tab3 = tk.Frame(tabs, bg="white"), tk.Frame(tabs, bg="white"), tk.Frame(tabs, bg="white")
# tabs.add(tab1, text=" REGISTRATION (UPLOAD) "); tabs.add(tab2, text=" SCANNER "); tabs.add(tab3, text=" RECORDS ")
# tabs.pack(expand=1, fill="both", padx=10, pady=10)

# # Tab 1
# tk.Label(tab1, text="STUDENT DETAILS", font=('Arial', 12, 'bold'), bg="white").pack(pady=10)
# txt_id = tk.Entry(tab1, font=('Arial', 12), width=30, bd=1, relief="solid"); txt_id.pack(pady=5)
# txt_name = tk.Entry(tab1, font=('Arial', 12), width=30, bd=1, relief="solid"); txt_name.pack(pady=5)
# tk.Button(tab1, text="STEP 1: UPLOAD PHOTO & SAVE", command=UploadAndRegister, bg="#4f46e5", fg="white", font=('Arial', 10, 'bold'), width=35, pady=12).pack(pady=20)
# tk.Button(tab1, text="STEP 2: TRAIN AI SYSTEM", command=TrainImages, bg="#16a34a", fg="white", font=('Arial', 10, 'bold'), width=35, pady=12).pack()
# lbl_status = tk.Label(tab1, text="Status: Ready", bg="white", fg="gray"); lbl_status.pack(pady=20)

# # Tab 2
# tk.Button(tab2, text="LAUNCH CAMERA SCANNER", command=TrackImages, bg="#4f46e5", fg="white", font=('Arial', 12, 'bold'), width=30, pady=30).pack(pady=100)

# # Tab 3
# tv = ttk.Treeview(tab3, columns=('id','name','date','time'), show='headings')
# for col, head in zip(['id','name','date','time'], ['ID', 'NAME', 'DATE', 'TIME']):
#     tv.heading(col, text=head); tv.column(col, width=150)
# tv.pack(fill="both", expand=True, padx=20, pady=20)

# LoadAttendance()
# window.mainloop()