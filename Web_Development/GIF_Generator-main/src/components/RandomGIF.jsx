import React from 'react'
import Spinner from './Spinner';
import useGIF from '../hooks/useGIF'

const RandomGIF = () => {
    const {loading,fetchGIF,gif} = useGIF();
    return (
        <div className='bg-green-400 w-1/2 rounded-xl border border-black flex flex-col items-center justify-center gap-y-6 p-4'>
            <h1 className='text-2xl font-bold'>A Random GIF</h1>
            {
                loading ? (<Spinner />) : (<img className='rounded-lg' src={gif} alt="Random GIF" />)
            }
            <button className='w-11/12 rounded-xl p-2 bg-white font-bold text-2xl' onClick={fetchGIF}>Generate</button>
        </div>
    )
}

export default RandomGIF
