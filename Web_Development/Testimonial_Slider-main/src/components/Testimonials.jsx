import React, { useState } from 'react'
import { FaAngleLeft, FaAngleRight, FaQuoteLeft, FaQuoteRight } from "react-icons/fa";

const Testimonials = ({ reviews }) => {
    const len = reviews.length;
    const [index, setIndex] = useState(0);
    function forwardHandler() {
        setIndex((index + 1) % len);
    }
    function backwardHandler() {
        setIndex((index - 1 + len) % len);
    }
    function surpriseMe() {
        setIndex(Math.floor(Math.random() *len));
    }
    return (
        <>  
            <div className="testimonial-container">
                <div className='image-container'>
                    <img width={'150px'} height={'150px'} src={reviews[index].image} style={{borderRadius:'50%'}} alt="" />
                    <div className=''></div>
                </div>
                <div className='info-container'>
                    <p style={{ fontSize: '1.5rem' ,fontWeight:'bold'}}>{reviews[index].name}</p>
                    <p style={{ color:'#D3BDFD' }} className='job'>{reviews[index].job}</p>
                </div>
                <FaQuoteLeft color='#BB83C0' />
                <div>
                    <p className='biodata'>{reviews[index].text}</p>
                </div>
                <FaQuoteRight color='#BB83C0' />
                <div className='direction-control-btns'>
                    <button style={{ backgroundColor: 'white', border: 'white',cursor:'pointer' }} onClick={backwardHandler}><FaAngleLeft color='#BB83C0' size={'30px'}/></button>
                    <button style={{ backgroundColor: 'white', border: 'white', cursor: 'pointer' }} onClick={forwardHandler}><FaAngleRight color='#BB83C0' size={'30px'} /></button>
                </div>
                <button className='surprise-me' onClick={surpriseMe}>
                    Surprise Me
                </button>
            </div>
        </>
    )
}

export default Testimonials
