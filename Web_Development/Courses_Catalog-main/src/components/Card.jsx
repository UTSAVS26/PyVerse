import React from 'react'
import { FcLike, FcLikePlaceholder } from "react-icons/fc"
import { toast } from 'react-toastify';

const Card = ({course,likedCourses,setLikedCourses}) => {
    // console.log(course)
    function clickHandler(){
        if(likedCourses.includes(course.id)){
            setLikedCourses((prev)=> prev.filter((cid)=>(cid!==course.id)));
            toast.warning("Liked Removed",{theme:'dark'});
        }
        else{
            setLikedCourses((prev) => [...prev, course.id]);
            toast.success("Liked Successfully", { theme: 'dark' });
        }
    }
  return (
    <>
        <div className='card'>
            <div className='card-img-container'>
                <img style={{width:'300px'}} src={course.image.url} alt="" />
                <div className='card-liked-btn-container'>
                      <button style={{border:'2px solid white',backgroundColor:'inherit',borderRadius:'50%',cursor:'pointer'}} onClick={clickHandler}>
                          {
                              likedCourses.includes(course.id) ?
                                  (<FcLike fontSize="1.75rem" />)
                                  : (<FcLikePlaceholder fontSize="1.75rem" />)
                          }
                    </button>
                </div>
            </div>
            <div className='card-content'>
                <p style={{color:'white',fontWeight:'bold',fontSize:'1.125rem'}}>{course.title}</p>
                  <p style={{ color: 'white', marginTop:'0.5rem' }}>
                    {
                        course.description.length > 100 ?
                            (course.description.substr(0, 100)) + "..." :
                            (course.description)
                    }
                </p>
            </div>
        </div>
    </>
  )
}

export default Card
