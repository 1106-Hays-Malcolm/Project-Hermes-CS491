#This is just a small UI to test the function using the built in puthon library Tkinter
#Initial build is isolated from LLM to make sure that it functions independty correctly first
# Two text widgets one for containing the response and one for containing the user input
# I will have the user input sent to terminal and the response will be the first word plus an incrementing number for each query that is sent
# Two buttons a send button and an exit(finish) button

import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import os    



#Timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
transcript_file = f"transcript_{timestamp}.txt"
# simulated llm response just a counter for now 
response_counter = 1

#Path for transcript files to be saved
transcript_file = os.path.join("LLM_Transcripts", f"transcript_{timestamp}.txt")

# end tkinter session when program ends
def end_session():
    print("Setup Complete.")
    root.destroy()





def send_query():
    global response_counter
    # make sure user selects a map
    if map_choice.get() == "":
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, "LLM: Please select a map before sending a query.\n")
        output_box.config(state=tk.DISABLED)
        output_box.yview(tk.END)
        return

    user_input = input_box.get("1.0", tk.END).strip()  
    # Pull first word
    first_word = user_input.split()[0] if user_input.strip() else ""

    # Generate response
    response = f"{first_word} {response_counter}"    
    
    #Update transcript text file
    with open(transcript_file, "a") as f:    
        f.write(f"User: {user_input}\n")    
        f.write(f"LLM: {response}\n")    

    # Insert respnse into response box
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"LLM: {response}\n")
    output_box.config(state=tk.DISABLED)

    # Make the box auto-scroll to the end
    output_box.yview(tk.END)

    #Increment response counter for each response
    response_counter += 1

    # Clear the user input box
    input_box.delete("1.0", tk.END)

# define the main tkinter window
root = tk.Tk()
root.title("LLM UI Test")
root.geometry("550x600")

# label above output box
output_label = tk.Label(root, text="LLM Response:")
output_label.pack(anchor="w", padx=10, pady=(10, 0))

# Box that contains LLM output 15 rows tall with wrap text and expands if window is resized
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)
output_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
output_box.config(state=tk.DISABLED)

# label above input box
input_label = tk.Label(root, text="User Query:")
input_label.pack(anchor="w", padx=10, pady=(0, 0))

# define the text entry box for user input 
#This has height of 10 rows and expands when window is resized
input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
input_box.pack(padx=10, pady=5, fill=tk.BOTH)   

# label above map choices
map_label = tk.Label(root, text="Pick your current Map:")
map_label.pack(anchor="w", padx=10, pady=(5, 0))

# only one map can be selected at a time
map_choice = tk.StringVar(value="")

map_frame = tk.Frame(root)
map_frame.pack(padx=10, pady=5, anchor="w")

map_one = tk.Radiobutton(map_frame, text="Act One", variable=map_choice, value="Act One")
map_one.pack(side=tk.LEFT, padx=5)

map_two = tk.Radiobutton(map_frame, text="Act Two", variable=map_choice, value="Act Two")
map_two.pack(side=tk.LEFT, padx=5)

map_three = tk.Radiobutton(map_frame, text="Act Three", variable=map_choice, value="Act Three")
map_three.pack(side=tk.LEFT, padx=5)

map_four = tk.Radiobutton(map_frame, text="Act Four", variable=map_choice, value="Act Four")
map_four.pack(side=tk.LEFT, padx=5)

# This widget has ttwo buttons a send and exit. Send to LLM and exit from program
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

send_button = tk.Button(button_frame, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=5)

end_button = tk.Button(button_frame, text="End Session", command=end_session)
end_button.pack(side=tk.LEFT, padx=5)

# Call to start widget
root.mainloop()