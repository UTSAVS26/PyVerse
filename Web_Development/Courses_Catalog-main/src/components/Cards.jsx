import React, { useState } from 'react'
import Card from './Card';
const Cards = ({ courses,category }) => {
    const [likedCourses,setLikedCourses]=useState([]);
    function getCourses(){
        if(category==="All"){
            let allCourses=[];
            Object.values(courses).forEach(arr=>{
                arr.forEach(course=>{
                    allCourses.push(course);
                })
            })
            // console.log(allCourses);
            return allCourses;
        }
        else{
            // console.log(courses[category]);
            return courses[category];
        }
    }
    return (
        <div className='cards'>
            {
                getCourses().map((course)=>{
                    return (<Card key={course.id} course={course} likedCourses={likedCourses} setLikedCourses={setLikedCourses}/>)
                })
            }
        </div>
    )
}

export default Cards
