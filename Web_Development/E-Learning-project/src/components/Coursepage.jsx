import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import courseData from "./data/Courses.json"; // Assuming your course data is stored here

const CoursePage = () => {
  const { id } = useParams();
  const decodedTitle = decodeURIComponent(id);

  const [courses, setCourses] = useState([]); // Declare hooks at the top level
  const course = courseData.find((course) => course.title === decodedTitle);

  useEffect(() => {
    setCourses(courseData); // Use the correct variable for JSON data
  }, []);

  if (!course) {
    return <h1>Course not found</h1>;
  }

  return (
    <div>
      <div className="course-page">
        {/* Embedded Video */}
        <div className="video-container">
          <iframe
            src={course.videoUrl}
            allow="autoplay; fullscreen"
            allowFullScreen
            title="Course Preview"
          ></iframe>
        </div>
        {/* Course Header */}
        <div className="course-header">
          <div className="course-title">
            <h1>{course.title}</h1>
            <h3>{course.description}</h3>
            <h4>Language: {course.language}</h4>
            <h4>Course Level: {course.level}</h4>
            <h4>Rating: {course.rating}/5</h4>
            <h4>Duration: {course.duration}</h4>
          </div>
        </div>
      </div>
      <div className="recommendation">
        <h1>Also View</h1>
        <div className="courses">
          {courses.slice(0, 3).map((course) => (
            <div className="course-card" key={course.id}>
              <img src={course.image} alt={course.title} />
              <h3>{course.title}</h3>
              <p>{course.description}</p>

              <h4>Course Level : {course.level}</h4>
              <h4>Language : {course.language}</h4>
              <h3>
                Price: Free
                {/* Price: ₹<del>{course.price + 1000}</del> ₹{course.price}
                                  /- */}
              </h3>
              <Link to={`/course/${course.title}`}>
                <button className="content-btn">Explore</button>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CoursePage;
