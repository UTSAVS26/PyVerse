import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import coursesData from "./data/Courses.json";

function Home() {
  const [courses, setCourses] = useState([]);
  useEffect(() => {
    setCourses(coursesData); // Load data from imported JSON file
  }, []);

  return (
    <div>
      {/* Introduction Section */}
      <section className="introduction">
        <div className="home">
          <div className="welcome">
            <h1>Welcome to Instructo</h1>
            <p>
              At <strong>Instructo</strong>, our mission is to empower students
              and learners by providing access to high-quality, free educational
              resources. Whether you're exploring data science, diving into
              artificial intelligence, we've got you covered.
            </p>
            <p>
              Our platform also features a{" "}
              <strong>Personalized Roadmap Generator</strong>—a unique tool
              designed to help you create a customized learning path based on
              your current skills, career goals, and aspirations. With this
              roadmap, you can stay focused, measure your progress, and achieve
              your dreams one step at a time.
            </p>
            <div className="home-buttons">
              <Link to="/courses">
                <button className="button">Explore Courses</button>
              </Link>
              <Link to="/Roadmap">
                <button className="button">Generate Your Roadmap</button>
              </Link>
            </div>
          </div>
          <div className="image">
            <img src="/Images/study.png" alt="img" />
          </div>
        </div>
      </section>

      {/* Trending Courses Section */}
      <section>
        <div className="data-card">
          <h1>Trending Courses</h1>
          <div className="courses">
            {courses.slice(0, 3).map((course) => (
              <div className="course-card" key={course.id}>
                <img
                  src={course.image} // Use course image or default
                  alt={course.title}
                />
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
          <div className="course-btn">
            <Link to="/courses">
              <button className="content-btn">View All Courses</button>
            </Link>
          </div>
        </div>
      </section>

      {/* About and Contact Section */}
      <section>
        <div className="two-column-layout">
          <div className="about">
            <h3>About Us</h3>
            <p>
              At Instructo, we are committed to fostering an environment of
              academic excellence and providing equal opportunities for learners
              across the globe. Our platform is designed to empower students
              with the knowledge, skills, and tools they need to succeed in
              their chosen fields.
            </p>
          </div>
          <div className="contact-form">
            <h3>Contact Us</h3>
            <form>
              <input type="text" name="name" placeholder="Your Name" required />
              <input
                type="email"
                name="email"
                placeholder="Your Email"
                required
              />
              <textarea
                name="message"
                placeholder="Your Message"
                rows="4"
                required
              ></textarea>
              <button type="submit">Submit</button>
            </form>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
