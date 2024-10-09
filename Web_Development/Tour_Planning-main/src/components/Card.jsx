import React, { useState } from 'react'

const Card = ({ tour, removeHandler }) => {
    const [readMore,SetreadMpre]=useState(true);
    const desc= readMore ? tour.info.substring(0,200):tour.info;
    return (
        <>
            <div className='card'>
                <img className='tour-img' src={tour.image} alt="" />
                <div className='tour-data'>
                    <h2 className='tour-price'>${tour.price}</h2>
                    <h2 className='tour-name'>{tour.name}</h2>
                    <p>{desc}
                        {tour.info.length >= 200 ? 
                            (<span className='readmore' onClick={()=>{
                                SetreadMpre(!readMore);
                            }}>{readMore ? "...Read More" : "Show Less"}</span>) : ("")
                        }
                    </p>
                </div>
                <button onClick={() => removeHandler(tour.id)} className='Isinterested'>
                    Not Interested
                </button>
            </div>
        </> 
    )
}

export default Card
