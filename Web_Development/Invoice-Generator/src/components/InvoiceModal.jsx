import React, { Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";
import { toPng } from "html-to-image";
import { jsPDF } from "jspdf";

const InvoiceModal = ({
    isOpen,
    setIsOpen,
    invoiceInfo,
    items,
    onAddNextInvoice,
}) => {
    function closeModal() {
        setIsOpen(false);
    }

    const addNextInvoiceHandler = () => {
        setIsOpen(false);
        onAddNextInvoice();
    };

    const SaveAsPDFHandler = () => {
        const dom = document.getElementById("print");
        toPng(dom)
            .then((dataUrl) => {
                const img = new Image();
                img.crossOrigin = "annoymous";
                img.src = dataUrl;
                img.onload = () => {
                    // Initialize the PDF.
                    const pdf = new jsPDF({
                        orientation: "portrait",
                        unit: "in",
                        format: [5.5, 8.5],
                    });

                    // Define reused data
                    const imgProps = pdf.getImageProperties(img);
                    const imageType = imgProps.fileType;
                    const pdfWidth = pdf.internal.pageSize.getWidth();

                    // Calculate the number of pages.
                    const pxFullHeight = imgProps.height;
                    const pxPageHeight = Math.floor(
                        (imgProps.width * 8.5) / 5.5
                    );
                    const nPages = Math.ceil(pxFullHeight / pxPageHeight);

                    // Define pageHeight separately so it can be trimmed on the final page.
                    let pageHeight = pdf.internal.pageSize.getHeight();

                    // Create a one-page canvas to split up the full image.
                    const pageCanvas = document.createElement("canvas");
                    const pageCtx = pageCanvas.getContext("2d");
                    pageCanvas.width = imgProps.width;
                    pageCanvas.height = pxPageHeight;

                    for (let page = 0; page < nPages; page++) {
                        // Trim the final page to reduce file size.
                        if (
                            page === nPages - 1 &&
                            pxFullHeight % pxPageHeight !== 0
                        ) {
                            pageCanvas.height = pxFullHeight % pxPageHeight;
                            pageHeight =
                                (pageCanvas.height * pdfWidth) /
                                pageCanvas.width;
                        }
                        // Display the page.
                        const w = pageCanvas.width;
                        const h = pageCanvas.height;
                        pageCtx.fillStyle = "white";
                        pageCtx.fillRect(0, 0, w, h);
                        pageCtx.drawImage(
                            img,
                            0,
                            page * pxPageHeight,
                            w,
                            h,
                            0,
                            0,
                            w,
                            h
                        );

                        // Add the page to the PDF.
                        if (page) pdf.addPage();

                        const imgData = pageCanvas.toDataURL(
                            `image/${imageType}`,
                            1
                        );
                        pdf.addImage(
                            imgData,
                            imageType,
                            0,
                            0,
                            pdfWidth,
                            pageHeight
                        );
                    }
                    // Output / Save
                    pdf.save(`invoice-${invoiceInfo.invoiceNumber}.pdf`);
                };
            })
            .catch((error) => {
                console.error("oops, something went wrong!", error);
            });
    };

    return (
        <Transition appear show={isOpen} as={Fragment}>
            <Dialog
                as="div"
                className="fixed inset-0 z-10 overflow-y-auto"
                onClose={closeModal}
            >
                <div className="min-h-screen px-4 text-center">
                    <Transition.Child
                        as={Fragment}
                        enter="ease-out duration-300"
                        enterFrom="opacity-0"
                        enterTo="opacity-100"
                        leave="ease-in duration-200"
                        leaveFrom="opacity-100"
                        leaveTo="opacity-0"
                    >
                        <Dialog.Overlay className="fixed inset-0 bg-black/50" />
                    </Transition.Child>

                    {/* This element is to trick the browser into centering the modal contents. */}
                    <span
                        className="inline-block h-screen align-middle"
                        aria-hidden="true"
                    >
                        &#8203;
                    </span>
                    <Transition.Child
                        as={Fragment}
                        enter="ease-out duration-300"
                        enterFrom="opacity-0 scale-95"
                        enterTo="opacity-100 scale-100"
                        leave="ease-in duration-200"
                        leaveFrom="opacity-100 scale-100"
                        leaveTo="opacity-0 scale-95"
                    >
                        <div className="my-8 inline-block w-full max-w-md transform overflow-hidden rounded-lg bg-white text-left align-middle shadow-xl transition-all">
                            <div className="p-4" id="print">
                                <h1 className="text-center text-lg font-bold text-gray-900">
                                    INVOICE
                                </h1>
                                <div className="mt-6">
                                    <div className="mb-4 grid grid-cols-2">
                                        <span className="font-bold">
                                            Invoice Number:
                                        </span>
                                        <span>{invoiceInfo.invoiceNumber}</span>
                                        <span className="font-bold">
                                            Cashier:
                                        </span>
                                        <span>{invoiceInfo.cashierName}</span>
                                        <span className="font-bold">
                                            Customer:
                                        </span>
                                        <span>{invoiceInfo.customerName}</span>
                                    </div>

                                    <table className="w-full text-left">
                                        <thead>
                                            <tr className="border-y border-black/10 text-sm md:text-base">
                                                <th>ITEM</th>
                                                <th className="text-center">
                                                    QTY
                                                </th>
                                                <th className="text-right">
                                                    PRICE
                                                </th>
                                                <th className="text-right">
                                                    AMOUNT
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {items.map((item) => (
                                                <tr key={item.id}>
                                                    <td className="w-full">
                                                        {item.name}
                                                    </td>
                                                    <td className="min-w-[50px] text-center">
                                                        {item.qty}
                                                    </td>
                                                    <td className="min-w-[80px] text-right">
                                                        $
                                                        {Number(
                                                            item.price
                                                        ).toFixed(2)}
                                                    </td>
                                                    <td className="min-w-[90px] text-right">
                                                        $
                                                        {Number(
                                                            item.price *
                                                                item.qty
                                                        ).toFixed(2)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>

                                    <div className="mt-4 flex flex-col items-end space-y-2">
                                        <div className="flex w-full justify-between border-t border-black/10 pt-2">
                                            <span className="font-bold">
                                                Subtotal:
                                            </span>
                                            <span>
                                                $
                                                {invoiceInfo.subtotal.toFixed(
                                                    2
                                                )}
                                            </span>
                                        </div>
                                        <div className="flex w-full justify-between">
                                            <span className="font-bold">
                                                Discount:
                                            </span>
                                            <span>
                                                $
                                                {invoiceInfo.discountRate.toFixed(
                                                    2
                                                )}
                                            </span>
                                        </div>
                                        <div className="flex w-full justify-between">
                                            <span className="font-bold">
                                                Tax:
                                            </span>
                                            <span>
                                                $
                                                {invoiceInfo.taxRate.toFixed(2)}
                                            </span>
                                        </div>
                                        <div className="flex w-full justify-between border-t border-black/10 py-2">
                                            <span className="font-bold">
                                                Total:
                                            </span>
                                            <span className="font-bold">
                                                $
                                                {invoiceInfo.total % 1 === 0
                                                    ? invoiceInfo.total
                                                    : invoiceInfo.total.toFixed(
                                                          2
                                                      )}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="mt-4 flex space-x-2 px-4 pb-6">
                                <button
                                    className="flex w-full items-center justify-center space-x-1 rounded-md border border-blue-500 py-2 text-sm text-blue-500 shadow-sm hover:bg-blue-500 hover:text-white"
                                    onClick={SaveAsPDFHandler}
                                >
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        className="h-4 w-4"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                                        />
                                    </svg>
                                    <span>Download</span>
                                </button>
                                <button
                                    onClick={addNextInvoiceHandler}
                                    className="flex w-full items-center justify-center space-x-1 rounded-md bg-blue-500 py-2 text-sm text-white shadow-sm hover:bg-blue-600"
                                >
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        className="h-4 w-4"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M13 5l7 7-7 7M5 5l7 7-7 7"
                                        />
                                    </svg>
                                    <span>Next</span>
                                </button>
                            </div>
                        </div>
                    </Transition.Child>
                </div>
            </Dialog>
        </Transition>
    );
};

export default InvoiceModal;
