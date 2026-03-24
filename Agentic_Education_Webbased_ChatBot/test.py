import app

# Initialize Google APIs - this updates the global services in the app module
app.init_google_apis_oauth()

# Now access the classroom_service through the app module
classroom_service = app.classroom_service
course_id = app.CLASSROOM_COURSE_ID

if classroom_service:
    try:
        courses = classroom_service.courses().list().execute()
        print(f"✅ Found {len(courses.get('courses', []))} courses")
        
        for course in courses.get("courses", []):
            print(f"  - {course['name']} (ID: {course['id']})")
    except Exception as e:
        print(f"❌ Error listing courses: {e}")
else:
    print("❌ classroom_service is None")