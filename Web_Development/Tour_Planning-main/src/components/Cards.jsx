import React from 'react'
import Card from './Card'

const Cards = ({tours,removeHandler}) => {
  return (
    <>
        <div className='cards'>
            {
            tours.map((tour)=>{
              return (<Card key={tour.id} tour={tour} removeHandler={removeHandler}/>)
            })
            }
        </div>
    </>
  )
}

export default Cards
