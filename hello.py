def process_message(sender, text):
    append_to_memory(sender, "user", text)
    current_state = user_states.get(sender)

    # --- Universal Cancel/Menu ---
    if text.lower().strip() in ['cancel', 'stop', 'menu', 'start', 'exit']:
        if current_state:
            del user_states[sender]; del user_temp_data[sender]
            send_whatsapp_message(sender, "Okay, I've canceled the current operation.")
        menu_text = "Hello! I'm your OS teaching assistant. Please choose an option:"
        options = ["Ask OS Question", "Create Worksheet", "Create PPT", "Upload Material", "Podcast from Image", "Summary from Image", "Create Video"]
        send_menu_message(sender, menu_text, options)
        append_to_memory(sender, "assistant", "Displayed main menu.")
        return

    # --- Initial Triggers for New Conversations (if no active state) ---
    if not current_state:
        if text.lower() == "create a worksheet":
            user_states[sender] = "awaiting_worksheet_topic"
            send_interactive_message(sender, "Let's create a worksheet! Please choose a topic:", ["CPU Scheduling", "Deadlocks", "Other Topics"])
            return
        if text.lower() == "create a ppt":
            user_states[sender] = "awaiting_ppt_topic"
            send_whatsapp_message(sender, "Excellent! What topic would you like the presentation to be about?")
            return
        if text.lower() == "ask an os question":
            user_states[sender] = "awaiting_os_question"
            send_whatsapp_message(sender, "Of course! What is your question about Operating Systems?")
            return
        if text.lower() == "upload material":
            user_states[sender] = "awaiting_material_pdf"
            send_whatsapp_message(sender, "Please send the PDF file you would like to upload as material.")
            return
        if text.lower() == "podcast from image":
            user_states[sender] = "awaiting_podcast_image"
            send_whatsapp_message(sender, "Please send me an image of the text you'd like to convert to a podcast.")
            return
        if text.lower() == "summary from image":
            user_states[sender] = "awaiting_summary_image"
            send_whatsapp_message(sender, "Please send me an image of the text you'd like me to summarize.")
            return
        if text.lower() == "create a video":
            send_whatsapp_message(sender, "This feature is coming soon!")
            end_conversation_and_show_menu(sender, None)
            return
    if current_state == "awaiting_os_question":
        response = query_os_subject(text)
        end_conversation_and_show_menu(sender, response["result"])
        return
    
    # --- State-Based Conversation Flow ---
    if current_state == "awaiting_ppt_topic":
        try:
            topic = text.strip()
            send_whatsapp_message(sender, f"Okay, generating a 10-slide presentation on '{topic}'. This may take a moment...")
            
            ppt_content = generate_ppt_content(topic)
            if ppt_content:
                ppt_bytes = create_ppt_file(topic, ppt_content)
                if ppt_bytes:
                    status = send_whatsapp_ppt(sender, ppt_bytes, f"{topic.replace(' ', '_')}.pptx")
                    if status != "success":
                        send_whatsapp_message(sender, "I created the PPT, but there was an error sending it.")
                else:
                    send_whatsapp_message(sender, "Sorry, I generated the content but failed to create the PPT file.")
            else:
                send_whatsapp_message(sender, "Sorry, I couldn't generate content for that topic.")
        except Exception as e:
             print(f"Error in PPT generation flow: {e}")
             send_whatsapp_message(sender, "An unexpected error occurred.")
        finally:
            user_states.pop(sender, None)
            user_temp_data.pop(sender, None)
        return

    # --- State-Based Conversation Flow ---
    if current_state == "awaiting_worksheet_topic":
        if text.lower() == "other topics":
            user_states[sender] = "awaiting_custom_topic"
            send_whatsapp_message(sender, "Please type the custom topic for your worksheet.")
            return
        user_temp_data[sender]['topic'] = text
        user_states[sender] = "awaiting_worksheet_format"
        send_interactive_message(sender, f"Great! Topic is '{text}'.\nWhat format would you like?", ["PDF Worksheet", "Google Form Quiz"])
        return

    if current_state == "awaiting_custom_topic":
        user_temp_data[sender]['topic'] = text
        user_states[sender] = "awaiting_worksheet_format"
        send_interactive_message(sender, f"Great! Topic is '{text}'.\nWhat format would you like?", ["PDF Worksheet", "Google Form Quiz"])
        return
        
    if current_state == "awaiting_worksheet_format":
        user_temp_data[sender]['format'] = text
        user_states[sender] = "awaiting_worksheet_quantity"
        send_interactive_message(sender, f"Perfect, a {text}.\nHow many questions?", ["5", "10", "15"])
        return

    if current_state == "awaiting_worksheet_quantity":
        try:
            user_temp_data[sender]['quantity'] = int(text.strip())
            if user_temp_data[sender]['format'] == "Google Form Quiz":
                user_temp_data[sender]['type'] = "mcq"
                topic = user_temp_data[sender]['topic']
                quantity = user_temp_data[sender]['quantity']
                send_whatsapp_message(sender, f"Okay! Generating a {quantity}-question Google Form quiz on '{topic}'. Please wait...")
                
                worksheet_content_result = generate_worksheet_content_text(topic, quantity, "mcq", user_memory[sender])
                if worksheet_content_result.get("source") == "generated_worksheet_text":
                    form_result = create_google_form_mcq(f"Quiz: {topic}", worksheet_content_result["result"])
                    send_whatsapp_message(sender, form_result["result"])
                else:
                    send_whatsapp_message(sender, "Sorry, I couldn't generate the quiz content.")
                
                del user_states[sender]; del user_temp_data[sender]
                return
            else:
                user_states[sender] = "awaiting_worksheet_type"
                send_interactive_message(sender, f"Perfect, {text} questions.\nNow, what type of questions?", ["MCQ", "Short Answer", "Numerical"])
        except ValueError:
            send_whatsapp_message(sender, "Please select a valid number from the buttons.")
        return

    if current_state == "awaiting_worksheet_type":
        try:
            topic = user_temp_data[sender]['topic']
            quantity = user_temp_data[sender]['quantity']
            q_type = text.lower().replace(" answer", "")
            valid_types = ["mcq", "short", "numerical"]
            if q_type not in valid_types:
                send_whatsapp_message(sender, "That's not a valid type."); return

            send_whatsapp_message(sender, f"Okay! Generating {quantity} {q_type} questions on '{topic}'. Please wait...")
            
            worksheet_content_result = generate_worksheet_content_text(topic, quantity, q_type, user_memory[sender])
            
            if worksheet_content_result and worksheet_content_result.get("source") == "generated_worksheet_text":
                full_content = worksheet_content_result["result"]
                questions_text, answers_text = (full_content.split("--- ANSWERS ---", 1) + ["No answer key generated."])[:2]
                user_temp_data[sender]['questions_text'] = questions_text.strip()
                user_temp_data[sender]['answers_text'] = answers_text.strip()
                
                if worksheet_pdf_bytes := create_pdf_locally(f"Worksheet: {topic.title()}", questions_text.strip()):
                    send_whatsapp_document(sender, worksheet_pdf_bytes, f"{topic.replace(' ', '_')}_worksheet.pdf")
                
                if answer_key_pdf_bytes := create_pdf_locally(f"Answer Key: {topic.title()}", answers_text.strip()):
                    send_whatsapp_document(sender, answer_key_pdf_bytes, f"{topic.replace(' ', '_')}_answers.pdf")

                user_states[sender] = "awaiting_classroom_post_choice"
                send_interactive_message(sender, "I've sent the PDFs. Post to Google Classroom?", ["Post Questions Only", "Post with Answers", "Don't Post"])
            else:
                send_whatsapp_message(sender, "Sorry, I couldn't generate the worksheet content.")
                del user_states[sender]; del user_temp_data[sender]
        except (ValueError, KeyError):
            send_whatsapp_message(sender, "An error occurred. Please start over.")
            if sender in user_states: del user_states[sender]
            if sender in user_temp_data: del user_temp_data[sender]
        return
        
    if current_state == "awaiting_classroom_post_choice":
        if text.lower() == "don't post":
            send_whatsapp_message(sender, "Okay, I won't post it to Classroom. Let me know if you need anything else!")
            del user_states[sender]; del user_temp_data[sender]
            return
        
        user_temp_data[sender]['post_choice'] = text
        user_states[sender] = "awaiting_classroom_title_choice"
        topic = user_temp_data[sender].get('topic', 'Worksheet')
        default_title = f"{topic} Tutorial"
        if len(default_title) > 20:
            default_title = f"{topic} Notes"
        send_interactive_message(sender, "Great! What should be the title?", [default_title, "Type Manually"])
        return

    if current_state == "awaiting_classroom_title_choice":
        if text.lower() == "type manually":
            user_states[sender] = "awaiting_custom_classroom_title"
            send_whatsapp_message(sender, "Please type the title for the announcement.")
            return
        else:
            handle_final_classroom_post(sender, text)
            return

    if current_state == "awaiting_custom_classroom_title":
        handle_final_classroom_post(sender, text.strip())
        return
        
    if current_state == "awaiting_video_topic":
        topic = text.strip()
        send_whatsapp_message(sender, f"Great! I will start generating a video about '{topic}'. This might take a few moments.")
        # In a real implementation, you would trigger your video generation functions here.
        del user_states[sender]; del user_temp_data[sender]
        return


    # --- Fallback to a General Query ---
    print(f"Handling as a general query: '{text}'")
    response = query_os_subject(text) # This should be your smart RAG/Gemini function
    send_whatsapp_message(sender, response["result"])
    append_to_memory(sender, "assistant", response["result"])