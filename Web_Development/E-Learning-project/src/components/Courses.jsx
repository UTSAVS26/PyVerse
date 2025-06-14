import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import coursesData from "./data/Courses.json"; // Import courses.json file

function Courses() {
  const [courses, setCourses] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  // Load courses from the JSON file
  useEffect(() => {
    setCourses(coursesData); // Load data from imported JSON file
  }, []);

  // Filter courses based on search query and selected category
  const filteredCourses = courses.filter(
    (course) =>
      course.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
      (selectedCategory === "All" || course.category === selectedCategory)
  );

  return (
    <>
      <div>
        <div className="container11">
          <div className="search-c">
            <div className="search-c1">
              <div className="search-course">
                <input
                  type="text"
                  id="course"
                  name="course"
                  placeholder="What do you want to learn"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <span className="search-icon">&#x1F50D;</span>
              </div>
            </div>
          </div>

          <div className="content">
            <section className="course-categories">
              <h2>Course Categories</h2>
              <div className="categories">
                <div className="cat-btn">
                  <button
                    className="category-btn"
                    onClick={() => setSelectedCategory("Programming")}
                  >
                    Programming
                  </button>
                  <button
                    className="category-btn"
                    onClick={() => setSelectedCategory("Web Development")}
                  >
                    Web Development
                  </button>
                  <button
                    className="category-btn"
                    onClick={() => setSelectedCategory("Data Science")}
                  >
                    Data Science & Machine Learning
                  </button>
                </div>
                <div className="showall">
                  <button
                    className="category-btn"
                    onClick={() => setSelectedCategory("All")}
                  >
                    Show All
                  </button>
                </div>
              </div>
            </section>

            <section className="popular-now">
              <div className="view-all">
                <h1>See Available Courses</h1>
              </div>
              <div className="courses">
                {filteredCourses.length > 0 ? (
                  filteredCourses.map((course) => (
                    <div className="course-card" key={course.id}>
                      <img
                        src={course.image} // Dynamically load images
                        alt={course.title}
                      />
                      <h3>{course.title}</h3>
                      <p>{course.description}</p>
                      <h4>Course Level: {course.level}</h4>
                      <h4>Language: {course.language}</h4>
                      <h3>Price: Free</h3>
                      <Link to={`/course/${course.title}`}>
                        <button className="content-btn">Explore</button>
                      </Link>
                    </div>
                  ))
                ) : (
                  <p>No courses available for this category.</p>
                )}
              </div>
              <div className="view-all">
                <button className="content-btn">View More</button>
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}

export default Courses;
